from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM

from gemq.utils import model_utils


def test_integrated_single_gpu_avoids_auto_offload(monkeypatch):
    model = Mock()
    model.to.return_value = model
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda _: SimpleNamespace(is_integrated=True))
    synchronize = Mock()
    monkeypatch.setattr(torch.cuda, "synchronize", synchronize)
    infer = Mock(side_effect=AssertionError("Auto offload must not run"))
    monkeypatch.setattr(model_utils, "infer_auto_device_map", infer)
    monkeypatch.setattr(model_utils, "get_balanced_memory", infer)
    monkeypatch.setattr(model_utils, "dispatch_model", infer)

    assert model_utils.dispatch_model_to_all_devices(model) is model
    model.to.assert_called_once_with("cuda:0")
    synchronize.assert_called_once_with()


@pytest.mark.parametrize("count,properties", [
    (1, SimpleNamespace(is_integrated=False)),
    (1, SimpleNamespace()),
    (2, SimpleNamespace(is_integrated=True)),
])
def test_other_devices_keep_existing_dispatch(monkeypatch, count, properties):
    model = Mock()
    dispatched = Mock()
    memory = {0: "16GiB", "cpu": "32GiB"}
    device_map = {"": 0}
    monkeypatch.setattr(torch.cuda, "device_count", lambda: count)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda _: properties)
    monkeypatch.setattr(torch.cuda, "synchronize", Mock())
    balanced = Mock(return_value=memory)
    infer = Mock(return_value=device_map)
    dispatch = Mock(return_value=dispatched)
    monkeypatch.setattr(model_utils, "get_balanced_memory", balanced)
    monkeypatch.setattr(model_utils, "infer_auto_device_map", infer)
    monkeypatch.setattr(model_utils, "dispatch_model", dispatch)

    assert model_utils.dispatch_model_to_all_devices(model) is dispatched
    model.to.assert_not_called()
    balanced.assert_called_once_with(model)
    assert infer.call_args.kwargs["max_memory"] is memory
    assert "Qwen3MoeDecoderLayer" in infer.call_args.kwargs["no_split_module_classes"]
    dispatch.assert_called_once_with(model, device_map=device_map)


@pytest.mark.cuda
def test_integrated_qwen3_forward_and_router_gradients():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    if torch.cuda.device_count() != 1 or not getattr(torch.cuda.get_device_properties(0), "is_integrated", False):
        pytest.skip("Requires a single integrated CUDA device")
    torch.manual_seed(0)
    config = Qwen3MoeConfig(
        vocab_size=128, hidden_size=128, intermediate_size=256,
        moe_intermediate_size=128, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, head_dim=32,
        num_experts=4, num_experts_per_tok=2, max_position_embeddings=32,
    )
    config._attn_implementation = "eager"
    config.use_cache = False
    model = Qwen3MoeForCausalLM(config).eval()
    inputs = torch.randint(0, config.vocab_size, (1, 8))
    before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    with torch.no_grad():
        expected = model(input_ids=inputs).logits
    model = model_utils.dispatch_model_to_all_devices(model)
    assert not hasattr(model, "hf_device_map")
    for name, parameter in model.named_parameters():
        assert parameter.device == torch.device("cuda:0")
        torch.testing.assert_close(parameter.cpu(), before[name], rtol=0, atol=0)
    with torch.no_grad():
        actual = model(input_ids=inputs.cuda()).logits.cpu()
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)
    routers = model_utils.get_router_params(model, "Qwen/Qwen3-30B-A3B")
    model.requires_grad_(False)
    for parameter in routers:
        parameter.requires_grad_(True)
    model.train()
    data = inputs.cuda()
    loss = model(input_ids=data, labels=data).loss
    loss.backward()
    assert torch.isfinite(loss)
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in routers)
    assert any(parameter.grad.abs().max() > 0 for parameter in routers)
    assert all(parameter.grad is None for parameter in model.parameters() if not parameter.requires_grad)
