import torch


class Qwen3MoeReconstructionCache:
    """Reuse unchanged experts while preserving the original FP16 addition order."""

    @torch.inference_mode()
    def __init__(self, moe_block, inputs, outputs, squared_gradients, batch_size):
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.moe_block = moe_block
        self.outputs = outputs
        self.squared_gradients = squared_gradients
        self.batches = []
        self.is_exact = True
        sequence_length, hidden_size = inputs.shape[1:]
        for start in range(0, len(inputs), batch_size):
            end = min(start + batch_size, len(inputs))
            hidden = inputs[start:end].reshape(-1, hidden_size)
            routing = torch.softmax(moe_block.gate(hidden), dim=1, dtype=torch.float)
            routing, selected = routing.topk(moe_block.top_k, dim=-1)
            if moe_block.norm_topk_prob:
                routing /= routing.sum(dim=-1, keepdim=True)
            routing = routing.to(hidden.dtype)
            contributions = hidden.new_zeros((*selected.shape, hidden_size))
            indices = []
            for expert, module in enumerate(moe_block.experts):
                slot, row = torch.where(selected.T == expert)
                indices.append((slot, row))
                if row.numel():
                    contributions[row, slot] = module(hidden[row]) * routing[row, slot, None]

            # Native Qwen3 adds experts in ascending ID order, rounding each addition.
            order = selected.argsort(dim=1)
            contributions = contributions.gather(1, order[..., None].expand_as(contributions))
            selected = selected.gather(1, order)
            reconstructed = torch.zeros_like(hidden)
            for slot in range(moe_block.top_k):
                reconstructed.add_(contributions[:, slot])
            if not torch.equal(reconstructed, outputs[start:end].reshape_as(hidden)):
                self.is_exact = False
                self.batches.clear()
                return
            self.batches.append((start, end, sequence_length, hidden, routing,
                                 indices, contributions, selected))

    @torch.inference_mode()
    def score(self, expert):
        if not self.is_exact:
            raise RuntimeError("Cached MoE output differs from the native output")
        sample_losses = []
        for start, end, sequence_length, hidden, routing, indices, contributions, selected in self.batches:
            slot, row = indices[expert]
            if not row.numel():
                sample_losses.append(hidden.new_zeros(end - start, dtype=torch.float64))
                continue
            updated = self.moe_block.experts[expert](hidden[row]) * routing[row, slot, None]
            sorted_slot = (selected[row] == expert).to(torch.int64).argmax(dim=1)
            selected_contributions = contributions[row].clone()
            selected_contributions[torch.arange(row.numel(), device=hidden.device), sorted_slot] = updated
            reconstructed = torch.zeros_like(updated)
            for position in range(self.moe_block.top_k):
                reconstructed.add_(selected_contributions[:, position])
            output = self.outputs[start:end].reshape_as(hidden)[row]
            weight = self.squared_gradients[start:end].reshape_as(hidden)[row]
            # Retain all token positions so the FP64 reduction layout is unchanged.
            errors = hidden.new_zeros((hidden.shape[0], hidden.shape[1]), dtype=torch.float64)
            errors[row] = weight * (output.double() - reconstructed.double()).square()
            sample_losses.append(errors.reshape(end - start, sequence_length, -1).flatten(1).sum(dim=1))
        return sum(torch.cat(sample_losses).tolist())
