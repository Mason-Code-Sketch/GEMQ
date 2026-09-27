"""Regression coverage for DeepSeek auxiliary-loss disabling."""

from types import SimpleNamespace
import unittest

import torch
import torch.nn as nn

from gemq.utils.model_utils import disable_deepseek_aux_loss


class _Gate(nn.Module):
    def __init__(self, alpha=0.001):
        super().__init__()
        self.alpha = alpha


class AddAuxiliaryLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, loss):
        assert loss.numel() == 1
        ctx.dtype = loss.dtype
        ctx.required_aux_loss = loss.requires_grad
        return x

    @staticmethod
    def backward(ctx, grad_output):
        grad_loss = None
        if ctx.required_aux_loss:
            grad_loss = torch.ones(1, dtype=ctx.dtype, device=grad_output.device)
        return grad_output, grad_loss


class _MoeMlp(nn.Module):
    def __init__(self, alpha=0.001):
        super().__init__()
        self.gate = _Gate(alpha)


class _DenseMlp(nn.Module):
    pass


class _Layer(nn.Module):
    def __init__(self, mlp):
        super().__init__()
        self.mlp = mlp


class _Model(nn.Module):
    def __init__(self, mlps):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([_Layer(mlp) for mlp in mlps])
        self.config = SimpleNamespace()


class DeepSeekAuxLossTest(unittest.TestCase):
    def test_disables_each_moe_gate_and_skips_dense_layers(self):
        model = _Model([_DenseMlp(), _MoeMlp(0.001), _MoeMlp(0.25)])

        disabled_gates = disable_deepseek_aux_loss(
            model, "deepseek-ai/DeepSeek-V2-Lite"
        )

        self.assertEqual(disabled_gates, 2)
        self.assertEqual(model.model.layers[1].mlp.gate.alpha, 0.0)
        self.assertEqual(model.model.layers[2].mlp.gate.alpha, 0.0)

        inputs = torch.ones(2, requires_grad=True)
        AddAuxiliaryLoss.apply(inputs, None).sum().backward()
        self.assertTrue(torch.equal(inputs.grad, torch.ones_like(inputs)))

    def test_rejects_an_unrecognized_deepseek_gate_layout(self):
        model = _Model([_DenseMlp(), nn.Module()])
        model.model.layers[1].mlp.gate = nn.Module()

        with self.assertRaisesRegex(RuntimeError, "does not expose alpha"):
            disable_deepseek_aux_loss(model, "deepseek-ai/DeepSeek-V2-Lite")

    def test_leaves_other_model_types_unchanged(self):
        model = _Model([_MoeMlp(0.125)])

        disabled_gates = disable_deepseek_aux_loss(
            model, "Qwen/Qwen1.5-MoE-A2.7B"
        )

        self.assertEqual(disabled_gates, 0)
        self.assertEqual(model.model.layers[0].mlp.gate.alpha, 0.125)


if __name__ == "__main__":
    unittest.main()
