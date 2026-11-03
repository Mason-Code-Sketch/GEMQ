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
    def __init__(self, block, inputs, reference, routing_metadata=None):
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
        down = self.linear(mid, experts.down_proj, self.offsets)
        self.contributions = down * self.routing[:, None]
        if "exact" not in metadata:
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
