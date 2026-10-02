import copy
import time
from types import SimpleNamespace
from unittest.mock import Mock
import weakref

import pytest
import torch
from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM

from gemq import quantize
from gemq.quantizers.rtn import RTNWeightQuantizer
from gemq.utils.model_utils import get_named_linears, get_router_params


MODEL_NAME = "Qwen/Qwen3-30B-A3B"


def make_model(device="cpu", dtype=torch.float32, seqlen=16, full_attention=False):
    torch.manual_seed(0)
    config = Qwen3MoeConfig(
        vocab_size=128, hidden_size=128, intermediate_size=256,
        moe_intermediate_size=128, num_hidden_layers=4,
        num_attention_heads=32 if full_attention else 4,
        num_key_value_heads=4 if full_attention else 2,
        head_dim=128 if full_attention else 32,
        num_experts=4, num_experts_per_tok=2, max_position_embeddings=seqlen,
        attention_dropout=0.0,
    )
    config._attn_implementation = "eager"
    config.use_cache = False
    model = Qwen3MoeForCausalLM(config).to(device=device, dtype=dtype).train()
    model.requires_grad_(False)
    for parameter in get_router_params(model, MODEL_NAME):
        parameter.requires_grad_(True)
    return model


def test_only_last_32_attention_blocks_are_wrapped_and_restored(monkeypatch):
    layers = [SimpleNamespace(self_attn=torch.nn.Linear(4, 4)) for _ in range(48)]
    monkeypatch.setattr(quantize, "get_blocks", lambda *_: layers)
    model = Mock()
    with quantize.checkpoint_router_attention(model, MODEL_NAME, 32) as count:
        assert count == 32
        assert ["forward" in layer.self_attn.__dict__ for layer in layers] == [False] * 16 + [True] * 32
    assert all("forward" not in layer.self_attn.__dict__ for layer in layers)


def test_attention_forward_is_restored_after_failure(monkeypatch):
    first = SimpleNamespace(self_attn=torch.nn.Linear(4, 4))
    last = SimpleNamespace(self_attn=torch.nn.Linear(4, 4))
    original = last.self_attn.forward
    last.self_attn.forward = original
    monkeypatch.setattr(quantize, "get_blocks", lambda *_: [first, last])
    with pytest.raises(RuntimeError, match="test failure"):
        with quantize.checkpoint_router_attention(Mock(), MODEL_NAME, 32):
            raise RuntimeError("test failure")
    assert last.self_attn.forward is original
    assert "forward" not in first.self_attn.__dict__


@pytest.mark.parametrize("device,dtype", [
    ("cpu", torch.float32),
    pytest.param("cuda", torch.bfloat16, marks=pytest.mark.cuda),
])
def test_checkpoint_preserves_loss_gradients_and_adamw_update(device, dtype):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    baseline = make_model(device, dtype)
    checkpointed = copy.deepcopy(baseline)
    data = torch.randint(0, 128, (1, 16), device=device)
    results = []
    saved_bytes = []
    for model, count in ((baseline, 0), (checkpointed, 32)):
        snapshots = {name: p.detach().clone() for name, p in model.named_parameters() if not p.requires_grad}
        routers = get_router_params(model, MODEL_NAME)
        optimizer = torch.optim.AdamW(routers, lr=1e-4, weight_decay=1e-4)
        sizes = []

        def save_tensor(tensor):
            sizes.append(tensor.numel() * tensor.element_size())
            return tensor

        with quantize.checkpoint_router_attention(model, MODEL_NAME, count):
            with torch.autograd.graph.saved_tensors_hooks(save_tensor, lambda tensor: tensor):
                loss = model(input_ids=data, labels=data).loss
            loss.backward()
            gradients = [p.grad.detach().clone() for p in routers]
            optimizer.step()
        results.append((loss.detach(), gradients, [p.detach().clone() for p in routers]))
        saved_bytes.append(sum(sizes))
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                assert parameter.grad is None
                torch.testing.assert_close(parameter, snapshots[name], rtol=0, atol=0)
        assert model.config._attn_implementation == "eager"
        assert all("forward" not in layer.self_attn.__dict__ for layer in model.model.layers)
    torch.testing.assert_close(results[0][0], results[1][0], rtol=0, atol=0)
    for expected, actual in zip(results[0][1], results[1][1]):
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-7)
    for expected, actual in zip(results[0][2], results[1][2]):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert saved_bytes[1] < saved_bytes[0]


@pytest.mark.cuda
def test_2048_token_attention_checkpoint_reduces_cuda_peak():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    peaks = []
    durations = []
    for count in (0, 32):
        model = make_model("cuda", torch.bfloat16, seqlen=2048, full_attention=True)
        data = torch.randint(0, 128, (1, 2048), device="cuda")
        optimizer = torch.optim.AdamW(get_router_params(model, MODEL_NAME), lr=1e-4)
        quantize.release_unused_memory()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        with quantize.checkpoint_router_attention(model, MODEL_NAME, count):
            loss = model(input_ids=data, labels=data).loss
            loss.backward()
            optimizer.step()
        torch.cuda.synchronize()
        durations.append(time.perf_counter() - start)
        peaks.append(torch.cuda.max_memory_allocated())
        assert torch.isfinite(loss)
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in get_router_params(model, MODEL_NAME))
        del model, data, loss, optimizer
        quantize.release_unused_memory()
    print(f"2048-token attention smoke: peak_bytes={peaks}, step_seconds={durations}")
    assert peaks[1] < peaks[0]


@pytest.mark.cuda
def test_router_ft_restores_forward_and_releases_gradients(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    model = make_model("cuda", torch.bfloat16)
    model.config.use_cache = True
    args = SimpleNamespace(model_name=MODEL_NAME, verbose=False, rft_lr=1e-4,
                           rft_wd=1e-4, rft_epochs=1, nsamples=2, rft_batch_size=1)
    batches = [(torch.randint(0, 128, (1, 16)), None) for _ in range(2)]
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda _: SimpleNamespace(is_integrated=True))
    quantize.finetune_routers(model, batches, args)
    assert model.config.use_cache is True
    assert all(p.grad is None for p in model.parameters())
    assert all("forward" not in layer.self_attn.__dict__ for layer in model.model.layers)
    assert next(model.parameters()).dtype == torch.bfloat16

    def fail_forward(*args, **kwargs):
        raise RuntimeError("test failure")

    monkeypatch.setattr(model, "forward", fail_forward)
    with pytest.raises(RuntimeError, match="test failure"):
        quantize.finetune_routers(model, batches, args)
    assert model.config.use_cache is True
    assert all(p.grad is None for p in model.parameters())
    assert all("forward" not in layer.self_attn.__dict__ for layer in model.model.layers)


def test_packing_releases_old_modules_only_after_validation(monkeypatch):
    model = torch.nn.Sequential(torch.nn.Linear(4, 4))
    reference = weakref.ref(model[0])
    quant_modules = {"0": model[0]}
    packed_states = {0: {"weight": torch.ones(4)}}

    def replace(model, model_name, modules, quant_weight):
        assert quant_weight
        assert reference() is modules["0"]
        model[0] = torch.nn.Linear(4, 4)

    def check(model, modules, args):
        assert reference() is modules["0"]
        assert reference() is not model[0]

    monkeypatch.setattr(quantize, "replace_linears", replace)
    monkeypatch.setattr(quantize, "check_packing", check)
    quantize.pack_quantized_model(model, quant_modules, packed_states, SimpleNamespace(model_name=MODEL_NAME))
    assert quant_modules == {}
    assert packed_states == {}
    assert reference() is None


@pytest.mark.cuda
@torch.no_grad()
def test_qwen3_packed_forward_survives_old_module_release():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    model = make_model("cuda", torch.float16).eval()
    args = SimpleNamespace(model_name=MODEL_NAME, mixed=False, expert_wbits=2,
                           attn_wbits=4, gate_wbits=16, dense_wbits=4)
    bit_cfg = quantize.build_alloc_cfg(model, args)
    modules = {}
    for layer_id, layer in enumerate(model.model.layers):
        for name, module in get_named_linears(layer).items():
            bits = bit_cfg[layer_id][name]
            if bits >= 16:
                continue
            quantizer = RTNWeightQuantizer(module.weight.data, nbits=bits, groupsize=128)
            codes, scales, zeros = quantizer.quantize()
            weight = quantizer.dequantize(codes, scales, zeros).reshape_as(module.weight)
            module.weight.copy_(weight)
            module.register_buffer("quant_scales", scales)
            module.register_buffer("quant_zeros", zeros)
            module.register_buffer("quant_nbits", torch.tensor(bits))
            module.register_buffer("quant_groupsize", torch.tensor(128))
            modules[f"{layer_id}.{name}"] = module
    del module, quantizer, weight, codes, scales, zeros
    references = [weakref.ref(module) for module in modules.values()]
    data = torch.randint(0, 128, (1, 16), device="cuda")
    expected = model(input_ids=data).logits
    quantize.pack_quantized_model(model, modules, {}, args)
    assert not modules
    assert all(reference() is None for reference in references)
    actual = model(input_ids=data).logits
    torch.testing.assert_close(actual, expected, rtol=2e-3, atol=2e-3)


def test_failed_packing_keeps_modules_for_diagnosis(monkeypatch):
    modules = {"0": torch.nn.Linear(4, 4)}
    packed_states = {0: {"weight": torch.ones(4)}}
    monkeypatch.setattr(quantize, "replace_linears", Mock())
    monkeypatch.setattr(quantize, "check_packing", Mock(side_effect=RuntimeError("packing mismatch")))
    with pytest.raises(RuntimeError, match="packing mismatch"):
        quantize.pack_quantized_model(Mock(), modules, packed_states, SimpleNamespace(model_name=MODEL_NAME))
    assert modules
    assert packed_states


def test_memory_cleanup_does_not_request_cuda_on_cpu(monkeypatch):
    collect = Mock()
    empty = Mock()
    monkeypatch.setattr(quantize.gc, "collect", collect)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "empty_cache", empty)
    quantize.release_unused_memory()
    collect.assert_called_once_with()
    empty.assert_not_called()
