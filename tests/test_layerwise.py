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
