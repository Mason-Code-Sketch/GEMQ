from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from gemq.inference.layerwise import layer_device_map


def test_contiguous_layer_assignment():
    model = SimpleNamespace(model=SimpleNamespace(layers=[None] * 27))
    mapping = layer_device_map(model, [f"cuda:{i}" for i in range(4)])
    assert [mapping[f"model.layers.{i}"] for i in range(27)] == (
        ["cuda:0"] * 7 + ["cuda:1"] * 7 + ["cuda:2"] * 7 + ["cuda:3"] * 6
    )
    assert mapping["model.embed_tokens"] == "cuda:0"
    assert mapping["model.norm"] == mapping["lm_head"] == "cuda:3"
    assert len(mapping) == 30


@pytest.mark.parametrize("model_type,layers", [
    ("deepseek_v2", 27), ("qwen2_moe", 24), ("qwen3_moe", 48), ("mixtral", 32),
])
@pytest.mark.parametrize("device_count", [1, 2, 4, 8])
def test_all_model_layouts_accept_configurable_device_counts(model_type, layers, device_count):
    model = SimpleNamespace(config=SimpleNamespace(model_type=model_type),
                            model=SimpleNamespace(layers=[None] * layers))
    devices = [f"cuda:{2 * index}" for index in range(device_count)]
    mapping = layer_device_map(model, devices)
    assigned = [mapping[f"model.layers.{index}"] for index in range(layers)]
    assert assigned == sorted(assigned, key=devices.index)
    assert set(assigned) == set(devices)
    assert max(assigned.count(device) for device in devices) - min(assigned.count(device) for device in devices) <= 1


def test_tied_embedding_and_head_remain_together():
    model = SimpleNamespace(config=SimpleNamespace(tie_word_embeddings=True),
                            model=SimpleNamespace(layers=[None] * 4))
    mapping = layer_device_map(model, ["cuda:0", "cuda:1"])
    assert mapping["lm_head"] == mapping["model.embed_tokens"] == "cuda:0"
    assert mapping["model.norm"] == "cuda:1"


@pytest.mark.parametrize("model_type", ["deepseek_v2", "qwen2_moe", "qwen3_moe", "mixtral"])
def test_entry_loads_supported_models_on_cpu_then_distributes(monkeypatch, model_type):
    import torch
    import transformers
    from gemq import eval_zeroshot as entry
    from gemq.inference import layerwise
    from gemq.utils import hf_loading

    config = SimpleNamespace(model_type=model_type)
    model = Mock(config=config)
    model.parameters.return_value = []
    loader = Mock(return_value=model)
    distribute = Mock(return_value=model)
    compatibility = Mock()
    monkeypatch.setattr(transformers.AutoConfig, "from_pretrained", Mock(return_value=config))
    monkeypatch.setattr(entry.AutoTokenizer, "from_pretrained", Mock(return_value="tokenizer"))
    monkeypatch.setattr(entry, "load_quantized_model", loader)
    monkeypatch.setattr(entry, "align_deepseek_softmax_scale", Mock())
    monkeypatch.setattr(entry, "prepare_backend", Mock(return_value="cached"))
    monkeypatch.setattr(layerwise, "distribute_packed_model", distribute)
    monkeypatch.setattr(hf_loading, "ensure_deepseek_v2_remote_code_compat", compatibility)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 4)
    args = entry.parse_args(["--model_path", "/unused", "--layer-devices", "cuda:1", "cuda:3"])
    assert entry.load_model(args) == (model, "tokenizer")
    assert loader.call_args.kwargs["device"] == "cpu"
    distribute.assert_called_once_with(model, ["cuda:1", "cuda:3"])
    assert compatibility.call_count == int(model_type == "deepseek_v2")


@pytest.mark.parametrize("model_type", ["deepseek_v2", "qwen2_moe", "qwen3_moe", "mixtral"])
@pytest.mark.parametrize("offset", [0.0, 1.0])
def test_layerwise_cache_checks_each_model_and_synchronizes_devices(monkeypatch, model_type, offset):
    import torch
    from gemq import eval_zeroshot as entry
    from gemq.inference import cached_linear

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = torch.nn.Embedding(16, 8)
            self.config = SimpleNamespace(model_type=model_type)
            self.model = SimpleNamespace(layers=[])
            self.offset = 0.0

        def get_input_embeddings(self):
            return self.embedding

        def forward(self, ids, use_cache=False):
            assert not use_cache
            return SimpleNamespace(logits=torch.arange(16, dtype=torch.float32).expand(*ids.shape, 16) + self.offset)

    model = Model()
    tokenizer = Mock(return_value=SimpleNamespace(input_ids=torch.tensor([[1, 2, 3]])))
    cache = Mock(side_effect=lambda model: setattr(model, "offset", offset))
    synchronize = Mock()
    monkeypatch.setattr(cached_linear, "cache_quantized_linears", cache)
    monkeypatch.setattr(torch.cuda, "synchronize", synchronize)
    args = entry.parse_args(["--model_path", "/unused", "--layer-devices", "cuda:1", "cuda:3"])
    if offset:
        with pytest.raises(RuntimeError, match="numerical check|preserve packed logits"):
            entry.prepare_backend(model, tokenizer, args)
    else:
        assert entry.prepare_backend(model, tokenizer, args) == "cached"
    cache.assert_called_once_with(model)
    assert {call.args[0] for call in synchronize.call_args_list} == {"cuda:1", "cuda:3"}


@pytest.mark.parametrize("devices", [[], ["cuda:0", "cuda:0"], ["cuda:0", "cuda:1", "cuda:2"]])
def test_invalid_layer_assignment(devices):
    model = SimpleNamespace(model=SimpleNamespace(layers=[None] * 2))
    with pytest.raises(ValueError):
        layer_device_map(model, devices)


def test_packed_metadata_moves_before_dispatch(monkeypatch):
    from gemq.inference.layerwise import distribute_packed_model
    from hqq.core.quantize import HQQLinear
    import accelerate

    packed = HQQLinear.__new__(HQQLinear)
    import torch
    torch.nn.Module.__init__(packed)
    packed.cuda = Mock()
    module = Mock()
    module.modules.return_value = [packed]
    model = SimpleNamespace(model=SimpleNamespace(layers=[None] * 2), get_submodule=Mock(return_value=module))
    dispatch = Mock(return_value=model)
    monkeypatch.setattr(accelerate, "dispatch_model", dispatch)
    assert distribute_packed_model(model, ["cuda:0", "cuda:1"]) is model
    assert packed.cuda.call_count == 5
    assert module.to.call_count == 5
    assert dispatch.call_args.kwargs["main_device"] == "cuda:0"
    assert dispatch.call_args.kwargs["device_map"] == layer_device_map(model, ["cuda:0", "cuda:1"])


@pytest.mark.parametrize("extra", [
    ["--is_fp"], ["--model_dtype", "bfloat16"], ["--backend", "pytorch"],
    ["--layer-devices", "cpu"], ["--layer-devices", "cuda:0", "cuda:0"],
])
def test_layerwise_cli_rejects_incompatible_settings(extra):
    from gemq.eval_zeroshot import parse_args
    with pytest.raises(SystemExit):
        parse_args(["--model_path", "/unused", "--layer-devices", "cuda:0", "cuda:1", *extra])


def test_layerwise_cli_preserves_evaluation_defaults():
    from gemq.eval_zeroshot import parse_args
    args = parse_args(["--model_path", "/unused", "--layer-devices", "cuda:0", "cuda:1", "cuda:2", "cuda:3"])
    assert args.layer_devices == ["cuda:0", "cuda:1", "cuda:2", "cuda:3"]
    assert args.model_dtype == "float16"
    assert args.backend == "auto"
    assert args.seed == 0
    assert args.max_batch_size == 8
