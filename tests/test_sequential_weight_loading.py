"""Check bounded loading preserves tensors, outputs and environment restoration."""
import os
from unittest.mock import patch
import pytest
import torch
from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM
from gemq.utils.hf_loading import sequential_weight_loading

@pytest.mark.parametrize("previous", [None, "0", "1"])
def test_environment_restored_on_error(previous):
    with patch.dict(os.environ):
        os.environ.pop("HF_DEACTIVATE_ASYNC_LOAD", None)
        if previous is not None:
            os.environ["HF_DEACTIVATE_ASYNC_LOAD"] = previous
        with pytest.raises(RuntimeError):
            with sequential_weight_loading():
                assert os.environ["HF_DEACTIVATE_ASYNC_LOAD"] == "1"
                raise RuntimeError("load failed")
        assert os.environ.get("HF_DEACTIVATE_ASYNC_LOAD") == previous

def test_real_loader_weights_outputs_and_gradients(tmp_path):
    torch.manual_seed(0)
    model = Qwen3MoeForCausalLM(Qwen3MoeConfig(
        vocab_size=32, hidden_size=16, intermediate_size=32,
        moe_intermediate_size=8, num_hidden_layers=2, num_attention_heads=2,
        num_key_value_heads=1, head_dim=8, num_experts=4,
        num_experts_per_tok=2, tie_word_embeddings=False,
    )).eval()
    model.save_pretrained(tmp_path, max_shard_size="10KB")
    with patch.dict(os.environ):
        os.environ["HF_DEACTIVATE_ASYNC_LOAD"] = "0"
        async_model = Qwen3MoeForCausalLM.from_pretrained(
            tmp_path, dtype=torch.float32, device_map="cpu", attn_implementation="eager").eval()
        with sequential_weight_loading():
            sync_model = Qwen3MoeForCausalLM.from_pretrained(
                tmp_path, dtype=torch.float32, device_map="cpu", attn_implementation="eager").eval()
        assert os.environ["HF_DEACTIVATE_ASYNC_LOAD"] == "0"
    assert async_model.state_dict().keys() == sync_model.state_dict().keys()
    for key, value in async_model.state_dict().items():
        assert torch.equal(value, sync_model.state_dict()[key]), key
    ids = torch.tensor([[1, 2, 3, 4]])
    outputs = []
    for instance in [async_model, sync_model]:
        result = instance(input_ids=ids, labels=ids, use_cache=False)
        outputs.append(result.logits.detach())
        result.loss.backward()
    assert torch.equal(*outputs)
    for (name, a), (_, b) in zip(async_model.named_parameters(), sync_model.named_parameters()):
        if a.grad is None:
            assert b.grad is None, name
        else:
            assert torch.equal(a.grad, b.grad), name


def test_gpu_budget_retains_explicit_cpu_destination():
    from types import SimpleNamespace
    from gemq.utils.hf_loading import layer_grads_max_memory
    with patch("gemq.utils.hf_loading.psutil.virtual_memory",
               return_value=SimpleNamespace(available=76 * 1024**3)):
        assert layer_grads_max_memory(40) == {0: 40 * 1024**3, "cpu": 76 * 1024**3}
        assert layer_grads_max_memory(None) is None
    with pytest.raises(ValueError):
        layer_grads_max_memory(0)
