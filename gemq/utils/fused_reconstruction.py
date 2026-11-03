"""Single-batch reconstruction scoring with the native grouped-MM operations."""
from collections import defaultdict

import torch


@torch.inference_mode()
def grouped_routing(block, inputs):
    """Small routing metadata only; no expert activations are kept here."""
    hidden = inputs.reshape(-1, inputs.shape[-1])
    _, routing, indices = block.gate(hidden)
    top_k = indices.shape[-1]
    ids, perm = torch.sort(indices.reshape(-1))
    histc_input = ids.int() if inputs.device.type == "cuda" else ids.float()
    counts = torch.histc(histc_input, bins=block.experts.num_experts,
                         min=0, max=block.experts.num_experts - 1)
    offsets = counts.cumsum(0, dtype=torch.int32)
    inverse = torch.empty_like(perm)
    inverse[perm] = torch.arange(perm.numel(), device=inputs.device)
    return {"top_k": top_k, "perm": perm, "routing": routing.reshape(-1)[perm],
            "offsets": offsets, "bounds": [0] + offsets.tolist(), "inverse": inverse}


class GroupedBatchContributions:
    """Retain only one batch; keep native routing, GEMMs and top-k reduction."""

    @torch.inference_mode()
    def __init__(self, block, inputs, reference, routing_metadata=None, bounded_workspace=False):
        from transformers.integrations.moe import _grouped_linear
        self.experts = block.experts
        self.linear = _grouped_linear
        experts = self.experts
        self.supported = (
            getattr(experts.config, "_experts_implementation", None) == "grouped_mm"
            and experts.has_gate and not experts.has_bias
            and experts.is_concatenated and not experts.is_transposed
            and not hasattr(block, "shared_expert")
        )
        self.exact = False
        if not self.supported:
            return
        self.shape = inputs.shape
        metadata = routing_metadata if routing_metadata is not None else grouped_routing(block, inputs)
        self.top_k = metadata["top_k"]
        self.perm = metadata["perm"]
        self.routing = metadata["routing"]
        self.offsets = metadata["offsets"]
        self.bounds = metadata["bounds"]
        self.inverse = metadata["inverse"]
        hidden = inputs.reshape(-1, inputs.shape[-1])
        self.hidden = hidden[self.perm // self.top_k]
        up = self.linear(self.hidden, experts.gate_up_proj, self.offsets)
        mid = experts._apply_gate(up)
        if bounded_workspace:
            del up
        down = self.linear(mid, experts.down_proj, self.offsets)
        if bounded_workspace:
            del mid
        self.contributions = down * self.routing[:, None]
        if bounded_workspace:
            del down
        if "exact" not in metadata:
            if bounded_workspace:
                expected = reference.reshape(-1, self.shape[-1])
                exact = True
                for start in range(0, len(expected), 128):
                    end = min(start + 128, len(expected))
                    ordered = self.contributions.index_select(
                        0, self.inverse[start * self.top_k:end * self.top_k])
                    result = ordered.view(end - start, self.top_k, self.shape[-1]).sum(1)
                    if not torch.equal(result, expected[start:end]):
                        exact = False
                    del ordered, result
                metadata["exact"] = exact
            else:
                metadata["exact"] = torch.equal(self.output(), reference)
        self.exact = metadata["exact"]

    def output(self):
        ordered = self.contributions[self.inverse]
        return ordered.view(-1, self.top_k, self.shape[-1]).sum(1).reshape(self.shape)

    @torch.inference_mode()
    def candidate_output(self, expert_id):
        start, end = self.bounds[expert_id:expert_id + 2]
        if start == end:
            return self.output()
        # Preserve the full expert weight array and group index. Other groups have
        # zero rows; the selected group retains its original M, K, N and dtype.
        offsets = torch.zeros_like(self.offsets)
        offsets[expert_id:] = end - start
        up = self.linear(self.hidden[start:end], self.experts.gate_up_proj, offsets)
        mid = self.experts._apply_gate(up)
        down = self.linear(mid, self.experts.down_proj, offsets)
        updated = down * self.routing[start:end, None]
        before = self.contributions[start:end].clone()
        try:
            self.contributions[start:end].copy_(updated)
            return self.output()
        finally:
            self.contributions[start:end].copy_(before)


@torch.inference_mode()
def streamed_fused_reconstruction_scores(block, inputs, reference, gradients,
                                          bit_cfg, batch_size, quantizer, progress=None):
    """Preserve sample-wise FP64 loss summation without a model-wide cache."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    experts = block.experts
    count = experts.gate_up_proj.shape[0]
    losses = {e: {} for e in range(count)}
    metadata = [grouped_routing(block, inputs[start:start + batch_size])
                for start in range(0, len(inputs), batch_size)]
    for expert_id in range(count):
        destinations = [experts.gate_up_proj[expert_id], experts.down_proj[expert_id]]
        originals = [tensor.detach().clone() for tensor in destinations]
        # Only this expert's candidates are retained, never a layer/model cache.
        candidates = {bit: [quantizer(w, nbits=bit).quantize() for w in originals]
                      for bit in bit_cfg}
        try:
            sample_losses = {b: [] for b in bit_cfg}
            for batch_index, start in enumerate(range(0, len(inputs), batch_size)):
                end = min(start + batch_size, len(inputs))
                x, y = inputs[start:end], reference[start:end]
                weights = gradients[start:end].to(x.device).double().pow(2)
                cache = GroupedBatchContributions(block, x, y, metadata[batch_index])
                for bitwidth in bit_cfg:
                    try:
                        for dst, quantized in zip(destinations, candidates[bitwidth]):
                            dst.copy_(quantized)
                        output = cache.candidate_output(expert_id) if cache.exact else block(x)
                        errors = weights * (y.double() - output.double()).pow(2)
                        sample_losses[bitwidth].append(errors.flatten(1).sum(1))
                        del output, errors
                    finally:
                        for dst, original in zip(destinations, originals):
                            dst.copy_(original)
                del cache, weights
            for bitwidth in bit_cfg:
                losses[expert_id][bitwidth] = sum(torch.cat(sample_losses[bitwidth]).tolist())
            del sample_losses
        finally:
            for dst, original in zip(destinations, originals):
                dst.copy_(original)
        del candidates, originals, destinations
        if progress is not None:
            progress(expert_id + 1, count)
    return defaultdict(dict, losses)


class DiskContributions:
    """Exact temporary outputs with explicit bounded residency of file pages."""

    def __init__(self, directory, shape, dtype):
        import tempfile
        from concurrent.futures import ThreadPoolExecutor
        import mmap
        import math
        import os
        self.os = os
        self.mmap_module = mmap
        self.shape = shape
        self.item_bytes = torch.empty((), dtype=dtype).element_size()
        self.size = math.prod(shape) * self.item_bytes
        self.file = tempfile.TemporaryFile(prefix=".re-output-cache-", dir=directory)
        self.file.truncate(self.size)
        self.pool = ThreadPoolExecutor(max_workers=8)
        os.posix_fadvise(self.file.fileno(), 0, 0, os.POSIX_FADV_RANDOM)
        self.mapping = mmap.mmap(self.file.fileno(), self.size)
        self.mapping.madvise(mmap.MADV_RANDOM)
        self.tensor = torch.frombuffer(self.mapping, dtype=dtype).reshape(shape)

    def drop(self, offset, size, write=False):
        page = self.mmap_module.PAGESIZE
        first = offset // page * page
        last = min(self.size, (offset + size + page - 1) // page * page)
        if write:
            self.mapping.flush(first, last - first)
        self.mapping.madvise(self.mmap_module.MADV_DONTNEED, first, last - first)
        self.os.posix_fadvise(self.file.fileno(), first, last - first,
                              self.os.POSIX_FADV_DONTNEED)

    def write_batch(self, index, cache, row_chunk=128):
        seq, top_k, hidden = self.shape[1:]
        row_bytes = top_k * hidden * self.item_bytes
        for start in range(0, seq, row_chunk):
            end = min(start + row_chunk, seq)
            ordered = cache.contributions.index_select(
                0, cache.inverse[start * top_k:end * top_k]).view(end - start, top_k, hidden)
            self.tensor[index, start:end].copy_(ordered)
            del ordered
            self.drop((index * seq + start) * row_bytes,
                      (end - start) * row_bytes, write=True)

    def read_rows(self, index, rows, device):
        seq, top_k, hidden = self.shape[1:]
        row_bytes = top_k * hidden * self.item_bytes
        selected = torch.empty((len(rows), top_k, hidden), dtype=self.tensor.dtype)
        raw = memoryview(selected.view(torch.uint8).numpy()).cast("B")
        def read_row(pair):
            target, row = pair
            target_start = target * row_bytes
            offset = (index * seq + row) * row_bytes
            destination = raw[target_start:target_start + row_bytes]
            done = 0
            while done < row_bytes:
                received = self.os.preadv(self.file.fileno(), [destination[done:]], offset + done)
                if not received:
                    raise EOFError("Incomplete anonymous reconstruction cache")
                done += received
        list(self.pool.map(read_row, enumerate(rows.tolist())))
        result = selected.to(device)
        del raw, selected
        self.drop(index * seq * row_bytes, seq * row_bytes)
        return result

    def close(self):
        self.pool.shutdown(wait=True)
        self.tensor = None
        self.mapping.close()
        self.file.close()


@torch.inference_mode()
def cross_expert_reconstruction_scores(block, inputs, reference, gradients,
                                       bit_cfg, batch_size, quantizer,
                                       progress=None, scratch_dir=None):
    """Reuse exact baseline outputs across experts without retaining them in RAM."""
    import mmap
    import os
    import shutil
    import tempfile
    if batch_size != 1:
        return streamed_fused_reconstruction_scores(block, inputs, reference, gradients,
            bit_cfg, batch_size, quantizer, progress)
    if not all(hasattr(mmap, name) for name in ["MADV_RANDOM", "MADV_DONTNEED"]):
        return streamed_fused_reconstruction_scores(block, inputs, reference, gradients,
            bit_cfg, batch_size, quantizer, progress)
    if not hasattr(os, "posix_fadvise"):
        return streamed_fused_reconstruction_scores(block, inputs, reference, gradients,
            bit_cfg, batch_size, quantizer, progress)
    count = block.experts.gate_up_proj.shape[0]
    metadata = [grouped_routing(block, inputs[i:i + 1]) for i in range(len(inputs))]
    top_k = metadata[0]["top_k"]
    shape = (len(inputs), inputs.shape[1], top_k, inputs.shape[-1])
    required = inputs.numel() * top_k * inputs.element_size()
    if shutil.disk_usage(scratch_dir or tempfile.gettempdir()).free < required + 1024**3:
        return streamed_fused_reconstruction_scores(block, inputs, reference, gradients,
            bit_cfg, batch_size, quantizer, progress)
    losses = {e: {b: [] for b in bit_cfg} for e in range(count)}
    storage = DiskContributions(scratch_dir, shape, inputs.dtype)
    usable = True
    try:
        for i in range(len(inputs)):
            # Non-finite zero-error terms must retain native inf*0/NaN behavior.
            if (not torch.isfinite(gradients[i]).all().item()
                    or not torch.isfinite(reference[i]).all().item()):
                usable = False
                break
            cache = GroupedBatchContributions(block, inputs[i:i + 1],
                                              reference[i:i + 1], metadata[i], bounded_workspace=True)
            if not cache.exact:
                usable = False
                del cache
                break
            storage.write_batch(i, cache)
            metadata[i]["rows_cpu"] = (metadata[i]["perm"] // top_k).cpu()
            del cache
        if not usable:
            return streamed_fused_reconstruction_scores(block, inputs, reference, gradients,
                bit_cfg, batch_size, quantizer, progress)
        experts = block.experts
        from transformers.integrations.moe import _grouped_linear
        for expert_id in range(count):
            destinations = [experts.gate_up_proj[expert_id], experts.down_proj[expert_id]]
            originals = [p.detach().clone() for p in destinations]
            candidates = {b: [quantizer(w, nbits=b).quantize() for w in originals]
                          for b in bit_cfg}
            try:
                for sample, meta in enumerate(metadata):
                    start, end = meta["bounds"][expert_id:expert_id + 2]
                    if start == end:
                        for b in bit_cfg:
                            losses[expert_id][b].append(inputs.new_zeros(1, dtype=torch.float64))
                        continue
                    rows_gpu = meta["perm"][start:end] // top_k
                    slots = meta["perm"][start:end] % top_k
                    rows_cpu = meta["rows_cpu"][start:end]
                    hidden = inputs[sample].index_select(0, rows_gpu)
                    offsets = torch.zeros_like(meta["offsets"])
                    offsets[expert_id:] = end - start
                    updates = {}
                    try:
                        for b in bit_cfg:
                            for dst, value in zip(destinations, candidates[b]):
                                dst.copy_(value)
                            up = _grouped_linear(hidden, experts.gate_up_proj, offsets)
                            mid = experts._apply_gate(up)
                            down = _grouped_linear(mid, experts.down_proj, offsets)
                            updates[b] = down * meta["routing"][start:end, None]
                            del up, mid, down
                    finally:
                        for dst, original in zip(destinations, originals):
                            dst.copy_(original)
                    errors = {b: inputs.new_empty((end - start, inputs.shape[-1]), dtype=torch.float64)
                              for b in bit_cfg}
                    for begin in range(0, len(rows_cpu), 128):
                        finish = min(begin + 128, len(rows_cpu))
                        cpu_rows, gpu_rows = rows_cpu[begin:finish], rows_gpu[begin:finish]
                        contributions = storage.read_rows(sample, cpu_rows, inputs.device)
                        weight = gradients[sample].index_select(0, cpu_rows).to(inputs.device).double().pow(2)
                        target = reference[sample].index_select(0, gpu_rows).double()
                        local_rows = torch.arange(finish - begin, device=inputs.device)
                        for b in bit_cfg:
                            contributions[local_rows, slots[begin:finish]] = updates[b][begin:finish]
                            output = contributions.sum(1)
                            value = weight * (target - output.double()).pow(2)
                            errors[b][begin:finish].copy_(value)
                            del output, value
                        del contributions, weight, target
                    full_error = inputs.new_zeros(inputs.shape[1:], dtype=torch.float64)
                    for b in bit_cfg:
                        full_error.index_copy_(0, rows_gpu, errors[b])
                        losses[expert_id][b].append(full_error.reshape(1, -1).sum(1))
                    del errors, full_error, updates, hidden
                for b in bit_cfg:
                    losses[expert_id][b] = sum(torch.cat(losses[expert_id][b]).tolist())
            finally:
                for dst, original in zip(destinations, originals):
                    dst.copy_(original)
            del originals, candidates, destinations
            if progress is not None:
                progress(expert_id + 1, count)
    finally:
        storage.close()
    return defaultdict(dict, losses)
