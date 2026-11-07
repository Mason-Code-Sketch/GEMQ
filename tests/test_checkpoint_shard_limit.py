"""Exercise the production save function without importing CUDA-only quantizers."""
import ast
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM

from gemq.utils.hf_loading import normalize_legacy_tied_weights_for_serialization


def production_save():
    source = Path(__file__).resolve().parents[1] / "gemq" / "quantize.py"
    module = ast.parse(source.read_text())
    function = next(node for node in module.body if isinstance(node, ast.FunctionDef)
                    and node.name == "save_quantized_model")
    hqq = SimpleNamespace(save_quantized=Mock())
    namespace = dict(torch=torch, AutoHQQHFModel=hqq, custom_object_save=Mock(),
                     normalize_legacy_tied_weights_for_serialization=
                     normalize_legacy_tied_weights_for_serialization)
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"), namespace)
    return namespace["save_quantized_model"], hqq


@pytest.mark.parametrize("save_dtype", ["float16", "bfloat16"])
def test_qwen3_round_trip_preserves_weights_and_limits_shards(tmp_path, save_dtype):
    torch.manual_seed(0)
    model = Qwen3MoeForCausalLM(Qwen3MoeConfig(
        vocab_size=32, hidden_size=16, intermediate_size=32,
        moe_intermediate_size=8, num_hidden_layers=2,
        num_attention_heads=2, num_key_value_heads=1, head_dim=8,
        num_experts=4, num_experts_per_tok=2, tie_word_embeddings=False,
    ))
    model.eval()
    dtype = getattr(torch, save_dtype)
    expected = {k: v.to(dtype).clone() for k, v in model.state_dict().items()}
    original = model.save_pretrained
    calls = []

    def record_save(path, **kwargs):
        calls.append(kwargs)
        return original(path, **kwargs)

    model.save_pretrained = record_save
    tokenizer = SimpleNamespace(save_pretrained=Mock())
    save, _ = production_save()
    save(model, tokenizer, str(tmp_path), save_dtype, real_quant=False)
    assert calls == [{"max_shard_size": "5GB", "save_original_format": False}]
    tokenizer.save_pretrained.assert_called_once_with(str(tmp_path))
    reloaded = Qwen3MoeForCausalLM.from_pretrained(tmp_path, dtype=dtype).eval()
    assert expected.keys() == reloaded.state_dict().keys()
    assert all(torch.equal(value, reloaded.state_dict()[key])
               for key, value in expected.items())
    assert not (tmp_path / "model.safetensors.index.json").exists()
    assert list(tmp_path.glob("*.safetensors"))


def test_packed_checkpoint_keeps_native_writer(tmp_path):
    save, hqq = production_save()
    model = SimpleNamespace(to=Mock(), save_pretrained=Mock())
    tokenizer = SimpleNamespace(save_pretrained=Mock())
    save(model, tokenizer, str(tmp_path), "float16", real_quant=True)
    hqq.save_quantized.assert_called_once_with(model, str(tmp_path))
    model.to.assert_not_called()
    model.save_pretrained.assert_not_called()


def test_save_error_is_not_silently_ignored(tmp_path):
    save, _ = production_save()
    model = torch.nn.Linear(2, 2)
    model.save_pretrained = Mock(side_effect=OSError("disk failure"))
    tokenizer = SimpleNamespace(save_pretrained=Mock())
    with pytest.raises(OSError, match="disk failure"):
        save(model, tokenizer, str(tmp_path), "float16", real_quant=False)
