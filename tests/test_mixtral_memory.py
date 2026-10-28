import copy
import pickle
from types import SimpleNamespace
import weakref

import pytest
import torch
from transformers import MixtralConfig, MixtralForCausalLM
from transformers.models.mixtral.modeling_mixtral import MixtralSparseMoeBlock

from gemq import compute_model_stats as stats, quantize
from gemq.quantizers.rtn import MCMoeRTNWeightQuantizer
from gemq.utils.model_utils import ModelType, get_router_params


MODEL_NAME = "mistralai/Mixtral-8x7B-v0.1"


def config():
    return MixtralConfig(
        vocab_size=128, hidden_size=128, intermediate_size=128,
        num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=2,
        num_local_experts=4, num_experts_per_tok=2, max_position_embeddings=32,
        attention_dropout=0.0, router_jitter_noise=0.0,
        use_cache=False, attn_implementation="eager",
    )


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.cuda)])
@torch.inference_mode()
def test_layer_re_restores_cpu_storage_with_exact_scores(tmp_path, monkeypatch, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    torch.manual_seed(29)
    template = MixtralForCausalLM(config()).to(dtype=torch.float16).eval()
    inputs = torch.randint(0, 128, (1, 4))
    gradient_path = tmp_path / "grads.pt"
    torch.save({i: torch.randn(2, 1, 4, 128, dtype=torch.float16)
                for i in range(3)}, gradient_path)
    if device == "cpu":
        base_tensor_to, module_to = torch.Tensor.to, torch.nn.Module.to

        def fake_tensor_to(tensor, *args, **kwargs):
            if args and args[0] == "cuda":
                args = ("cpu",) + args[1:]
            if kwargs.get("device") == "cuda":
                kwargs["device"] = "cpu"
            return base_tensor_to(tensor, *args, **kwargs)

        def fake_module_to(module, *args, **kwargs):
            if args and args[0] == "cuda":
                return module._apply(lambda tensor: tensor.clone())
            return module_to(module, *args, **kwargs)

        monkeypatch.setattr(torch.Tensor, "to", fake_tensor_to)
        monkeypatch.setattr(torch.nn.Module, "to", fake_module_to)
        monkeypatch.setattr(stats, "move_embed", lambda *_: None)

    protected = set()
    tensor_to, load = torch.Tensor.to, torch.load

    def checked_to(tensor, *args, **kwargs):
        target = kwargs.get("device", args[0] if args else None)
        if tensor.device.type == "cpu" and str(target).startswith("cuda"):
            assert tensor.untyped_storage().data_ptr() not in protected
        return tensor_to(tensor, *args, **kwargs)

    def tracked_load(*args, **kwargs):
        values = load(*args, **kwargs)
        if protected:
            protected.update(tensor.untyped_storage().data_ptr() for tensor in values.values())
        return values

    monkeypatch.setattr(torch.Tensor, "to", checked_to)
    monkeypatch.setattr(torch, "load", tracked_load)
    results = []
    for cached in (False, True):
        model = copy.deepcopy(template)
        before = [{name: tensor for name, tensor in layer.state_dict().items()}
                  for layer in model.model.layers]
        protected.clear()
        if cached:
            protected.update(tensor.untyped_storage().data_ptr()
                             for state in before for tensor in state.values())
        path = tmp_path / f"scores-{cached}.pkl"
        args = SimpleNamespace(
            model_name=MODEL_NAME, layer_grads_path=str(gradient_path),
            layer_re_path=str(path), forward_batch_size=1, wbits="1,2,3",
            cache_mixtral_layer_grads=cached,
        )
        stats.compute_faster_layer_re(model, [(inputs, None)] * 2, args)
        with path.open("rb") as stream:
            results.append(pickle.load(stream))
        for layer, original in zip(model.model.layers, before):
            for name, tensor in layer.state_dict().items():
                assert tensor.device.type == "cpu"
                assert torch.equal(tensor, original[name])
                if cached:
                    assert tensor.data_ptr() == original[name].data_ptr()
    assert results[0] == results[1]


def legacy_scores(block, inputs, outputs, gradients, batch_size):
    # Eagerly retain every candidate, as the old implementation did.
    if hasattr(block.experts, "gate_up_proj"):
        experts = [[block.experts.gate_up_proj[i], block.experts.down_proj[i]]
                   for i in range(block.experts.num_experts)]
    else:
        experts = [[e.w1.weight, e.w2.weight, e.w3.weight] for e in block.experts]
    originals = [[w.clone() for w in weights] for weights in experts]
    quantizers = {
        (e, bit, j): MCMoeRTNWeightQuantizer(weight, bit)
        for e, weights in enumerate(originals)
        for bit in (1, 2, 3) for j, weight in enumerate(weights)
    }
    squared = gradients.double().pow(2).to(inputs.device)
    scores = {}
    for e, weights in enumerate(experts):
        scores[e] = {}
        for bit in (1, 2, 3):
            for j, weight in enumerate(weights):
                weight.copy_(quantizers[e, bit, j].quantize())
            scores[e][bit] = stats.compute_moe_reconstruction_error(
                block, inputs, outputs, squared, batch_size,
            )
            for weight, original in zip(weights, originals[e]):
                weight.copy_(original)
    return scores


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("cache_gradients", [False, True])
@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.cuda)])
@torch.inference_mode()
def test_streamed_candidates_and_mapped_gradients_match_legacy(tmp_path, monkeypatch, dtype, batch_size, cache_gradients, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    torch.manual_seed(13)
    block = MixtralForCausalLM(config()).model.layers[0].mlp.to(device=device, dtype=dtype).eval()
    inputs = torch.randn(3, 5, 128, dtype=dtype, device=device)
    outputs = torch.cat([stats.get_decoder_hidden_states(block(part), part.shape)
                         for part in inputs.split(batch_size)])
    path = tmp_path / "gradients.pt"
    torch.save(torch.randn_like(inputs).cpu(), path)
    gradients = torch.load(path, mmap=True, weights_only=True)
    expected = legacy_scores(copy.deepcopy(block), inputs, outputs, gradients, batch_size)
    originals = {name: parameter.clone() for name, parameter in block.named_parameters()}
    pointers = {name: parameter.data_ptr() for name, parameter in block.named_parameters()}
    references, live_counts, gradient_sizes = [], [], []
    double = torch.Tensor.double

    class TrackedQuantizer(MCMoeRTNWeightQuantizer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            references.append(weakref.ref(self))
            live_counts.append(sum(reference() is not None for reference in references))

    def tracked_double(tensor, *args, **kwargs):
        if tensor.untyped_storage().data_ptr() == gradients.untyped_storage().data_ptr():
            gradient_sizes.append(tensor.numel())
        return double(tensor, *args, **kwargs)

    monkeypatch.setattr(stats, "MCMoeRTNWeightQuantizer", TrackedQuantizer)
    monkeypatch.setattr(torch.Tensor, "double", tracked_double)
    actual = stats.compute_mixtral_layer_reconstruction_errors(
        block, inputs, outputs, gradients, [1, 2, 3], batch_size,
        cache_squared_gradients=cache_gradients,
    )
    assert actual == expected
    assert max(live_counts) == 1
    assert all(reference() is None for reference in references)
    if cache_gradients:
        assert not gradient_sizes
    else:
        assert gradient_sizes and max(gradient_sizes) <= batch_size * inputs[0].numel()
    for name, parameter in block.named_parameters():
        assert torch.equal(parameter, originals[name])
        assert parameter.data_ptr() == pointers[name]


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.cuda)])
def test_cached_half_gradient_squares_are_exact_and_do_not_alias(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    values = torch.arange(65536, dtype=torch.int32).to(torch.int16).view(torch.float16)
    values = values[torch.isfinite(values)]
    original = values.clone()
    expected = values.double().pow(2).to(device)
    actual = values.to(device=device).to(dtype=torch.float64, copy=True)
    actual.square_()
    assert torch.equal(actual, expected)
    assert torch.equal(values, original)


@pytest.mark.parametrize("failure", ["quantize", "score"])
@pytest.mark.parametrize("cache_gradients", [False, True])
@torch.inference_mode()
def test_candidate_failure_restores_original_weights(monkeypatch, failure, cache_gradients):
    block = MixtralForCausalLM(config()).model.layers[0].mlp.eval()
    originals = {name: parameter.clone() for name, parameter in block.named_parameters()}
    inputs = torch.randn(1, 3, 128)
    calls = 0

    def fail(*args, **kwargs):
        nonlocal calls
        calls += 1
        if failure == "score" or calls == 2:
            raise RuntimeError("test failure")
        return torch.zeros_like(args[0].x)

    if failure == "quantize":
        monkeypatch.setattr(MCMoeRTNWeightQuantizer, "quantize", fail)
    else:
        monkeypatch.setattr(stats, "compute_moe_reconstruction_error", fail)
    with pytest.raises(RuntimeError, match="test failure"):
        stats.compute_mixtral_layer_reconstruction_errors(
            block, inputs, inputs, inputs, [1, 2, 3], 1,
            cache_squared_gradients=cache_gradients,
        )
    for name, parameter in block.named_parameters():
        assert torch.equal(parameter, originals[name])


def test_mixtral_checkpoints_preserve_loss_gradients_updates_and_frozen_weights():
    torch.manual_seed(7)
    dtype = torch.bfloat16
    baseline = MixtralForCausalLM(config()).to(dtype=dtype).train()
    models = [baseline, copy.deepcopy(baseline)]
    inputs = torch.randint(0, 128, (1, 8))
    results, saved_bytes = [], []
    for model, count in zip(models, (0, 32)):
        model.requires_grad_(False)
        routers = get_router_params(model, MODEL_NAME)
        for parameter in routers:
            parameter.requires_grad_(True)
        frozen = {name: p.clone() for name, p in model.named_parameters() if not p.requires_grad}
        optimizer = torch.optim.AdamW(routers, lr=1e-4, weight_decay=1e-4)
        sizes = []

        def save(tensor):
            sizes.append(tensor.numel() * tensor.element_size())
            return tensor

        with quantize.checkpoint_router_attention(model, MODEL_NAME, count):
            with torch.autograd.graph.saved_tensors_hooks(save, lambda tensor: tensor):
                loss = model(input_ids=inputs, labels=inputs).loss
            loss.backward()
            gradients = [p.grad.clone() for p in routers]
            optimizer.step()
        assert torch.isfinite(loss)
        assert all(torch.isfinite(value).all() for value in gradients + list(routers))
        results.append((loss.detach(), gradients, [p.detach().clone() for p in routers]))
        saved_bytes.append(sum(sizes))
        for name, parameter in model.named_parameters():
            if name in frozen:
                assert parameter.grad is None and torch.equal(parameter, frozen[name])
        assert all("forward" not in layer.self_attn.__dict__ for layer in model.model.layers)
        assert next(model.parameters()).dtype == dtype
    assert torch.equal(results[0][0], results[1][0])
    for expected, actual in zip(results[0][1] + results[0][2], results[1][1] + results[1][2]):
        assert torch.equal(actual, expected)
    assert saved_bytes[1] < saved_bytes[0]


@pytest.mark.parametrize("override,integrated,expected", [(None, True, 32), (0, True, 0), (None, False, 0)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_mixtral_layer_grads_checkpoint_policy_and_values(tmp_path, monkeypatch, override, integrated, expected, dtype):
    torch.manual_seed(3)
    model = MixtralForCausalLM(config()).to(dtype=dtype).eval()
    inputs = torch.randint(0, 128, (1, 8))
    paths = [tmp_path / "reference.pt", tmp_path / "actual.pt"]
    args = SimpleNamespace(model_name=MODEL_NAME, layer_grads_path=str(paths[0]), attention_checkpoint_layers=0)
    flags = [parameter.requires_grad for parameter in model.parameters()]
    stats.compute_layer_grads(model, [(inputs, None)], args)
    checkpoint = stats.checkpoint_attention
    counts = []

    def capture(layers, count):
        counts.append(count)
        return checkpoint(layers, count)

    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda _: SimpleNamespace(is_integrated=integrated))
    monkeypatch.setattr(stats, "checkpoint_attention", capture)
    args.layer_grads_path, args.attention_checkpoint_layers = str(paths[1]), override
    stats.compute_layer_grads(model, [(inputs, None)], args)
    assert counts == [expected]
    reference, actual = [torch.load(path, weights_only=True, mmap=True) for path in paths]
    assert all(torch.equal(actual[layer], reference[layer]) for layer in reference)
    assert all(torch.isfinite(value).all() for value in actual.values())
    assert [parameter.requires_grad for parameter in model.parameters()] == flags
    assert all(not layer._backward_hooks and "forward" not in layer.self_attn.__dict__ for layer in model.model.layers)


@pytest.mark.parametrize("integrated,expected", [(True, 32), (False, 0)])
def test_mixtral_router_ft_auto_checkpoint_and_dtype_restore(monkeypatch, integrated, expected):
    model = MixtralForCausalLM(config()).to(dtype=torch.float16)
    checkpoint = quantize.checkpoint_router_attention
    counts = []

    def capture(model, name, count):
        counts.append(count)
        return checkpoint(model, name, count)

    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda _: SimpleNamespace(is_integrated=integrated))
    monkeypatch.setattr(quantize, "checkpoint_router_attention", capture)
    quantize.finetune_routers(model, [(torch.ones(1, 8, dtype=torch.long), None)], SimpleNamespace(
        model_name=MODEL_NAME, rft_lr=1e-4, rft_wd=1e-4, rft_epochs=0, verbose=False,
    ))
    assert counts == [expected]
    assert next(model.parameters()).dtype == torch.float16
    assert all(parameter.grad is None for parameter in model.parameters())
    assert all("forward" not in layer.self_attn.__dict__ for layer in model.model.layers)


@pytest.mark.parametrize("mcmoe", [False, True])
@torch.inference_mode()
def test_gptq_releases_finished_quantizers_without_changing_weights(monkeypatch, mcmoe):
    class Layer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.w1 = torch.nn.Linear(128, 128, bias=False, dtype=torch.float16)
            self.w2 = torch.nn.Linear(128, 128, bias=False, dtype=torch.float16)

        def forward(self, inputs):
            return inputs + self.w2(self.w1(inputs).relu())

        def to(self, device):
            return self if device == "cuda" else super().to(device)

    torch.manual_seed(5)
    layer = Layer()
    inputs = torch.randn(2, 4, 128, dtype=torch.float16)
    args = SimpleNamespace(
        model_name=MODEL_NAME, groupsize=128, blocksize=128, percdamp=0.01,
        actorder=False, static_groups=False, mse=True, reproduce_mcmoe=mcmoe,
        nsamples=2, verbose=False,
    )
    monkeypatch.setattr(quantize, "compute_decoder_inputs", lambda *_: (inputs.clone(), {}))
    monkeypatch.setattr(quantize, "get_blocks", lambda model, _: model.layers)
    monkeypatch.setattr(quantize, "build_alloc_cfg", lambda *_: {0: {"w1": 2, "w2": 2}})
    quantizer_name = "MCMoeGPTQWeightQuantizer" if mcmoe else "GPTQWeightQuantizer"
    original_quantizer = getattr(quantize, quantizer_name)
    results, lifetimes = [], []
    for model_type in (ModelType.OLMOE, ModelType.MIXTRAL):
        references, live_counts = [], []

        class TrackedQuantizer(original_quantizer):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                references.append(weakref.ref(self))

            def quantize(self):
                live_counts.append(sum(reference() is not None for reference in references))
                return super().quantize()

        monkeypatch.setattr(quantize, quantizer_name, TrackedQuantizer)
        monkeypatch.setitem(quantize.NAME_TO_MODEL, MODEL_NAME, model_type)
        model = SimpleNamespace(config=SimpleNamespace(use_cache=True), layers=[copy.deepcopy(layer)])
        modules, packed = quantize.quantize_weights_gptq(model, [], args)
        assert len(modules) == 2 and not packed and model.config.use_cache is True
        assert all(reference() is None for reference in references)
        results.append(model.layers[0].state_dict())
        lifetimes.append(live_counts)
    assert lifetimes == [[2, 2], [2, 1]]
    assert results[0].keys() == results[1].keys()
    assert all(torch.equal(results[0][name], results[1][name]) for name in results[0])
