"""Native Transformers 5 Mixtral scoring, GPTQ, Router and checkpoint regression."""
from functools import partial
from types import SimpleNamespace

import pytest
import torch
from hqq.models.hf.base import AutoHQQHFModel
from transformers import MixtralConfig, MixtralForCausalLM

from gemq import quantize, compute_model_stats as stats
from gemq.inference.fused_moe import HQQFusedMoeExperts, replace_fused_moe_experts
from gemq.inference.grouped_cached import install_grouped_cached
from gemq.utils.hf_loading import load_quantized_model
from gemq.utils.model_utils import get_moe_block, get_router_params, compute_gate_stats_hook_mixtral

NAME = "mistralai/Mixtral-8x7B-v0.1"


def make_model(device="cpu"):
    torch.manual_seed(19)
    config = MixtralConfig(vocab_size=128, hidden_size=128, intermediate_size=128,
        num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2,
        num_local_experts=3, num_experts_per_tok=2, max_position_embeddings=32,
        attention_dropout=0.0, router_jitter_noise=0.0, use_cache=False)
    config._attn_implementation = "eager"
    return MixtralForCausalLM(config).to(device).eval()


@torch.no_grad()
def test_native_scoring_preserves_weights_and_counts():
    model = make_model()
    block = get_moe_block(model.model.layers[0], NAME)
    before = {k: v.clone() for k, v in block.state_dict().items()}
    x = torch.randn(2, 4, 128)
    inputs, outputs, weights, counts = [], [], [], []
    hook = block.register_forward_hook(partial(compute_gate_stats_hook_mixtral,
        inps=inputs, outs=outputs, weights=weights, counts=counts))
    y = block(x)
    hook.remove()
    assert outputs[0].shape == y.shape
    assert counts[0].sum() == 16 and counts[0].numel() == 3
    result = stats.compute_fused_moe_layer_reconstruction_errors(block, x, y,
        torch.ones_like(y, dtype=torch.float64), [1, 2, 3], 1)
    assert set(result) == {0, 1, 2}
    assert all(torch.isfinite(torch.tensor(v)) for row in result.values() for v in row.values())
    mc = stats.compute_fused_moe_quantization_losses(block, [x], [y], [2])
    assert set(mc) == {0, 1, 2}
    assert all(torch.equal(v, before[k]) for k, v in block.state_dict().items())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA HQQ packing")
def test_gptq_router_checkpoint_grouped_roundtrip(tmp_path):
    model = make_model("cuda").half()
    args = SimpleNamespace(model_name=NAME, mixed=False, expert_wbits=2,
        attn_wbits=4, gate_wbits=16, dense_wbits=4, groupsize=128, blocksize=128,
        percdamp=.01, actorder=False, static_groups=False, mse=True,
        reproduce_mcmoe=False, nsamples=2, verbose=False, real_quant=True,
        rft_lr=1e-4, rft_wd=1e-4, rft_epochs=1, rft_batch_size=1)
    data = [(torch.randint(0, 128, (1, 8), device="cuda"), None) for _ in range(2)]
    modules, packed = quantize.quantize_weights_gptq(model, data, args)
    assert len(packed[0]) == 3
    model.cuda()
    routers = get_router_params(model, NAME)
    assert len(routers) == 1
    router_before = routers[0].detach().clone()
    quantize.finetune_routers(model, data, args)
    assert not torch.equal(routers[0], router_before)
    model.requires_grad_(False).eval()
    with torch.no_grad():
        expected = model(data[0][0]).logits
        quantize.replace_linears(model, NAME, modules, quant_weight=True)
        replace_fused_moe_experts(model, packed)
        actual = model(data[0][0]).logits
        torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)
        AutoHQQHFModel.save_quantized(model, str(tmp_path))
        loaded = load_quantized_model(str(tmp_path), compute_dtype=torch.float16,
                                      device="cuda", trust_remote_code=False).eval()
        assert isinstance(loaded.model.layers[0].mlp.experts, HQQFusedMoeExperts)
        logits = loaded(data[0][0]).logits
        torch.testing.assert_close(logits, actual, rtol=2e-3, atol=2e-3)
        install_grouped_cached(loaded)
        grouped = loaded(data[0][0]).logits
        torch.testing.assert_close(grouped, logits, rtol=2e-3, atol=2e-3)
        assert torch.isfinite(loaded(data[0][0], labels=data[0][0]).loss)
