"""Native layout tests; CPU reference kernels do not certify Triton numerics."""
import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeSparseMoeBlock

from gemq.inference.fused_moe import HQQFusedMoeExperts
from gemq.inference import packed_fused
from gemq.inference.packed_fused import PackedExpertLinear, PackedFusedMoeExperts


def reference_gemm(x, wq, scales, zeros, bits, group_size):
    k = torch.arange(x.shape[1], device=x.device)
    q = ((wq[k // (32 // bits)].long() >> ((k % (32 // bits)) * bits)[:, None])
         & ((1 << bits) - 1)).half()
    w = ((q - zeros[k // group_size]).half() * scales[k // group_size]).half()
    return x @ w


class PackedFixture(nn.Module):
    """CPU fixture with GemLite's public packed layout, no CUDA implementation."""
    def __init__(self, width, output, bits):
        super().__init__()
        self.in_features, self.out_features = width, output
        self.W_nbits, self.group_size, self.W_group_mode = bits, 128, 3
        self.channel_scale_mode = 0
        self.compute_dtype, self.bias = torch.float16, None
        codes = torch.randint(0, 1 << bits, (width, output))
        size = 32 // bits
        packed = torch.zeros(((width + size - 1) // size, output), dtype=torch.int64)
        for k in range(width):
            packed[k // size] |= codes[k] << ((k % size) * bits)
        self.register_buffer('W_q', packed.to(torch.int32))
        self.register_buffer('scales', (torch.rand(width // 128, output) * .03 + .01).half())
        self.register_buffer('zeros', (torch.rand(width // 128, output) * ((1 << bits) - 1)).half())

    def forward(self, x):
        return reference_gemm(x, self.W_q, self.scales, self.zeros, self.W_nbits, self.group_size)


def config():
    cfg = Qwen3MoeConfig(hidden_size=128, intermediate_size=128,
        moe_intermediate_size=128, num_experts=4, num_experts_per_tok=2,
        num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2,
        head_dim=32, vocab_size=32, max_position_embeddings=32, use_cache=False)
    cfg._attn_implementation = 'eager'
    return cfg


def fixture_experts(cfg):
    experts = HQQFusedMoeExperts(cfg, device='cpu')
    for i, bits in enumerate((1, 2, 3, 4)):
        experts.gate_up_proj[i] = PackedFixture(128, 256, bits)
        experts.down_proj[i] = PackedFixture(128, 128, bits)
    return experts


@pytest.mark.parametrize('bits', [1, 2, 3, 4])
def test_projection_borrows_storage_and_keeps_combined_output(monkeypatch, bits):
    monkeypatch.setattr(packed_fused, '_packed_gemm', reference_gemm)
    torch.manual_seed(0)
    old = PackedFixture(128, 256, bits)
    new = PackedExpertLinear(old)
    for name in ('W_q', 'scales', 'zeros'):
        assert getattr(new, name) is getattr(old, name)
    x = torch.randn(7, 128).half()
    assert torch.equal(new(x), old(x))


@pytest.mark.parametrize('case', ['bf16', 'fma', 'bad_stride', 'bias'])
def test_rejects_unsupported_kernel_contract(case):
    old = PackedFixture(128, 256, 3)
    if case == 'bf16':
        old.compute_dtype = torch.bfloat16
    elif case == 'fma':
        old.W_group_mode = 4
    elif case == 'bad_stride':
        old.zeros = old.zeros.repeat(2, 1)[::2]
    else:
        old.bias = torch.zeros(256).half()
    with pytest.raises(ValueError):
        PackedExpertLinear(old)


@pytest.mark.parametrize('tokens', [1, 7])
def test_native_model_logits_loss_and_patch_entry(monkeypatch, tokens):
    monkeypatch.setattr(packed_fused, '_packed_gemm', reference_gemm)
    torch.manual_seed(42)
    cfg = config()
    model = Qwen3MoeForCausalLM(cfg).half().eval()
    block = model.model.layers[0].mlp
    block.experts = fixture_experts(cfg)
    ids = torch.randint(0, 32, (1, tokens))
    with torch.inference_mode():
        before = model(ids, labels=ids)
    gate = block.gate
    old = block.experts
    # Execute the real installer body without importing CUDA-only dependencies.
    path = Path(__file__).parents[1] / 'gemq/inference/patch.py'
    tree = ast.parse(path.read_text())
    installer = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                     and n.name == 'replace_moe_blocks')
    kind = SimpleNamespace(QWEN3MOE=4, MIXTRAL=1, DEEPSEEKV2=2, OLMOE=3)
    ns = dict(ModelType=kind, NAME_TO_MODEL={'qwen': 4},
              get_blocks=lambda m, _: m.model.layers,
              Qwen3MoeSparseMoeBlock=Qwen3MoeSparseMoeBlock)
    exec(compile(ast.Module(body=[installer], type_ignores=[]), str(path), 'exec'), ns)
    ns['replace_moe_blocks'](model, 'qwen', cfg)
    assert model.model.layers[0].mlp is block
    assert block.gate is gate
    assert isinstance(block.experts, PackedFusedMoeExperts)
    for old_p, new_p in zip(old.gate_up_proj, block.experts.gate_up_proj):
        assert new_p.W_q is old_p.W_q
    with torch.inference_mode():
        after = model(ids, labels=ids)
    assert torch.equal(before.logits, after.logits)
    if tokens > 1:
        assert torch.equal(before.loss, after.loss)


def test_rejected_conversion_preserves_source():
    old = fixture_experts(config())
    old.down_proj[-1].compute_dtype = torch.bfloat16
    state = {name: value.clone() for name, value in old.state_dict().items()}
    with pytest.raises(ValueError):
        PackedFusedMoeExperts(old)
    assert all(torch.equal(value, state[name]) for name, value in old.state_dict().items())


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires an idle CUDA device')
def test_actual_hqq_to_packed_native_model_strict_equivalence(monkeypatch):
    from gemq.inference.patch import prepare_for_inference
    from gemq.utils.quant_utils import create_hqq_linear_from_quantized_weights
    from gemq.triton_kernels.dequant_gemm import dequant_gemm_kernel

    def reject_tuning(*args, **kwargs):
        raise AssertionError('Native packed experts must not autotune per token count')
    monkeypatch.setattr(dequant_gemm_kernel, 'run', reject_tuning)

    torch.manual_seed(42)
    cfg = config()
    model = Qwen3MoeForCausalLM(cfg).cuda().half().eval()
    block = model.model.layers[0].mlp
    experts = HQQFusedMoeExperts(cfg)
    for i, bits in enumerate((1, 2, 3, 4)):
        for name, output in [('gate_up_proj', 256), ('down_proj', 128)]:
            codes = torch.randint(0, 1 << bits, (output, 128), device='cuda').half()
            scales = (torch.rand(output, 1, device='cuda') * .03 + .01).half()
            zeros = (torch.rand(output, 1, device='cuda') * ((1 << bits) - 1)).half()
            getattr(experts, name)[i] = create_hqq_linear_from_quantized_weights(
                codes, scales, zeros, (output, 128), bits, 128, device='cuda')
    block.experts = experts
    ids = torch.randint(0, 32, (1, 8), device='cuda')
    with torch.inference_mode():
        before = model(ids, labels=ids)
        prepare_for_inference(model, 'Qwen/Qwen3-30B-A3B')
        after = model(ids, labels=ids)
    assert torch.equal(before.logits, after.logits), 'Triton logits differ from original HQQ'
    assert torch.equal(before.loss, after.loss), 'Triton loss differs from original HQQ'


def test_single_group_channel_metadata_preserves_scales(monkeypatch):
    monkeypatch.setattr(packed_fused, '_packed_gemm', reference_gemm)
    old = PackedFixture(128, 256, 3)
    old.W_group_mode, old.channel_scale_mode = 1, 1
    new = PackedExpertLinear(old)
    x = torch.randn(5, 128).half()
    assert new.scales is old.scales and new.zeros is old.zeros
    assert torch.equal(new(x), old(x))


def test_native_prepare_converts_only_experts(monkeypatch):
    import gc
    cfg = config()
    model = Qwen3MoeForCausalLM(cfg).half().eval()
    model.model.layers[0].mlp.experts = fixture_experts(cfg)
    path = Path(__file__).parents[1] / 'gemq/inference/patch.py'
    tree = ast.parse(path.read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                    and n.name == 'prepare_for_inference')
    converted, installed = [], []
    ns = dict(get_blocks=lambda m, _: m.model.layers,
              replace_linear_recursive=converted.append,
              replace_moe_blocks=lambda *a, **kw: installed.append(a[0]),
              gc=gc, torch=torch)
    monkeypatch.setattr(torch.cuda, 'empty_cache', lambda: None)
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), 'exec'), ns)
    assert ns['prepare_for_inference'](model, 'qwen') is model
    assert converted == [model.model.layers[0].mlp.experts]
    assert installed == [model]
