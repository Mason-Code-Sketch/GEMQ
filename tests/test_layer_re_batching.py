import math

import pytest
import torch

from gemq.compute_model_stats import compute_moe_reconstruction_error


class ToyMoe(torch.nn.Module):
    def __init__(self, tuple_output):
        super().__init__()
        self.tuple_output = tuple_output
        self.calls = 0

    def forward(self, inputs):
        self.calls += 1
        output = inputs * 1.125
        return (output, None) if self.tuple_output else output


@pytest.mark.parametrize("tuple_output", [False, True])
@pytest.mark.parametrize("batch_size", [1, 2, 4, 8])
def test_layer_re_matches_original_sample_reduction(tuple_output, batch_size):
    generator = torch.Generator().manual_seed(41)
    inputs = torch.randn(5, 7, 11, generator=generator, dtype=torch.float64)
    outputs = torch.randn(5, 7, 11, generator=generator, dtype=torch.float64)
    weights = torch.rand(5, 7, 11, generator=generator, dtype=torch.float64)
    expected = sum(
        (weights[i:i+1] * (outputs[i:i+1] - inputs[i:i+1] * 1.125).pow(2)).sum().item()
        for i in range(len(inputs))
    )
    module = ToyMoe(tuple_output)
    actual = compute_moe_reconstruction_error(
        module, inputs, outputs, weights, batch_size
    )
    assert actual == pytest.approx(expected, rel=1e-14, abs=1e-14)
    assert module.calls == math.ceil(len(inputs) / batch_size)


@pytest.mark.parametrize("batch_size", [0, -1])
def test_layer_re_rejects_invalid_batch_size(batch_size):
    inputs = torch.ones(1, 2, 3)
    with pytest.raises(ValueError, match="positive"):
        compute_moe_reconstruction_error(ToyMoe(False), inputs, inputs, inputs, batch_size)
