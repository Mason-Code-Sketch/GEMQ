from types import SimpleNamespace

import pytest
import torch
from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM

from gemq import compute_model_stats


MODEL_NAME = "Qwen/Qwen3-30B-A3B"


def make_model(device="cpu", dtype=torch.float32):
    torch.manual_seed(0)
    config = Qwen3MoeConfig(
        vocab_size=128,
        hidden_size=128,
        intermediate_size=256,
        moe_intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        num_experts=4,
        num_experts_per_tok=2,
        max_position_embeddings=32,
    )
    config._attn_implementation = "eager"
    return Qwen3MoeForCausalLM(config).to(device=device, dtype=dtype)


def make_batches():
    generator = torch.Generator().manual_seed(1)
    return [(torch.randint(0, 128, (1, 8), generator=generator), None) for _ in range(4)]


def reference_gradients(model, batches):
    gradients = {i: [] for i in range(len(model.model.layers))}
    handles = []
    for i, layer in enumerate(model.model.layers):
        def save_gradient(module, inputs, outputs, layer_id=i):
            gradients[layer_id].append(outputs[0].detach().cpu().clone())

        handles.append(layer.register_full_backward_hook(save_gradient))
    model.config.use_cache = False
    try:
        for data, _ in batches:
            model(input_ids=data.to(model.device), labels=data.to(model.device)).loss.backward()
    finally:
        for handle in handles:
            handle.remove()
    return {i: torch.stack(values) for i, values in gradients.items()}


@pytest.mark.parametrize("device,dtype", [
    ("cpu", torch.float32),
    pytest.param("cuda", torch.float32, marks=pytest.mark.cuda),
    pytest.param("cuda", torch.float16, marks=pytest.mark.cuda),
])
def test_streamed_gradients_match_full_parameter_backward(tmp_path, monkeypatch, device, dtype):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    model = make_model(device, dtype)
    model.train()
    model.get_input_embeddings().weight.requires_grad_(False)
    batches = make_batches()
    expected = reference_gradients(model, batches)
    original_flags = [parameter.requires_grad for parameter in model.parameters()]
    model.config.use_cache = True
    path = tmp_path / "LayerGrads.pt"
    original_save = torch.save

    def check_file_backed_buffers(gradients, filename):
        assert all(value.untyped_storage().filename for value in gradients.values())
        assert all(parameter.grad is None for parameter in model.parameters())
        original_save(gradients, filename)

    monkeypatch.setattr(torch, "save", check_file_backed_buffers)
    compute_model_stats.compute_layer_grads(
        model, batches, SimpleNamespace(model_name=MODEL_NAME, layer_grads_path=str(path))
    )
    actual = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    for i in expected:
        torch.testing.assert_close(actual[i], expected[i], rtol=0, atol=0)
    assert [parameter.requires_grad for parameter in model.parameters()] == original_flags
    assert model.config.use_cache is True
    assert list(tmp_path.iterdir()) == [path]
    assert not model.get_input_embeddings()._forward_hooks
    assert all(not layer._backward_hooks for layer in model.model.layers)


def test_failed_backward_preserves_cache_and_restores_model(tmp_path, monkeypatch):
    model = make_model()
    original_flags = [parameter.requires_grad for parameter in model.parameters()]
    original_forward = model.forward
    calls = 0

    def fail_second_batch(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("test failure")
        return original_forward(*args, **kwargs)

    monkeypatch.setattr(model, "forward", fail_second_batch)
    path = tmp_path / "LayerGrads.pt"
    path.write_bytes(b"previous cache")
    with pytest.raises(RuntimeError, match="test failure"):
        compute_model_stats.compute_layer_grads(
            model, make_batches(), SimpleNamespace(model_name=MODEL_NAME, layer_grads_path=str(path))
        )
    assert path.read_bytes() == b"previous cache"
    assert list(tmp_path.iterdir()) == [path]
    assert [parameter.requires_grad for parameter in model.parameters()] == original_flags
    assert model.config.use_cache is True
    assert not model.get_input_embeddings()._forward_hooks
    assert all(not layer._backward_hooks for layer in model.model.layers)


@pytest.mark.cuda
def test_mapped_layer_re_matches_eager_loading_and_batching(tmp_path, monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    model = make_model("cuda")
    batches = make_batches()
    path = tmp_path / "LayerGrads.pt"
    args = SimpleNamespace(
        model_name=MODEL_NAME,
        layer_grads_path=str(path),
        layer_re_path=str(tmp_path / "LayerRE.pkl"),
        wbits="1,2,3",
        forward_batch_size=1,
    )
    compute_model_stats.compute_layer_grads(model, batches, args)
    monkeypatch.setattr(compute_model_stats, "get_all_expert_names", lambda name: [
        f"mlp.experts.{i}" for i in range(4)
    ])
    model.cpu().eval()
    original_load = torch.load
    results = []
    for mmap, batch_size in ((False, 1), (True, 1), (True, 2)):
        def load_gradients(*load_args, **kwargs):
            assert kwargs["mmap"] is True
            kwargs["mmap"] = mmap
            return original_load(*load_args, **kwargs)

        monkeypatch.setattr(torch, "load", load_gradients)
        args.forward_batch_size = batch_size
        compute_model_stats.compute_faster_layer_re(model, batches, args)
        import pickle
        with open(args.layer_re_path, "rb") as file:
            results.append(pickle.load(file))
    for scores in results[1:]:
        for layer_id in results[0]:
            for expert_id in results[0][layer_id]:
                for bitwidth in (1, 2, 3):
                    assert scores[layer_id][expert_id][bitwidth] == pytest.approx(
                        results[0][layer_id][expert_id][bitwidth], rel=1e-5, abs=1e-12
                    )
