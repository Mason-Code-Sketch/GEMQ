"""Real HF serialization round trips for fused MoE checkpoints."""
import json
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch
from transformers import MixtralConfig, MixtralForCausalLM, Qwen3MoeConfig, Qwen3MoeForCausalLM
from test_checkpoint_shard_limit import production_save


@pytest.mark.parametrize('family', ['mixtral', 'qwen3'])
@pytest.mark.parametrize('dtype', ['float16', 'bfloat16'])
def test_native_fused_shards_round_trip_without_reverse_conversion(tmp_path, family, dtype):
    torch.manual_seed(0)
    common=dict(vocab_size=32, hidden_size=16, intermediate_size=32,
                num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
                tie_word_embeddings=False)
    if family=='mixtral':
        model=MixtralForCausalLM(MixtralConfig(**common, num_local_experts=4, num_experts_per_tok=2))
    else:
        model=Qwen3MoeForCausalLM(Qwen3MoeConfig(**common, moe_intermediate_size=8,
                                              num_experts=4, num_experts_per_tok=2, head_dim=8))
    model.to(getattr(torch,dtype)).eval()
    expected={k:v.clone() for k,v in model.state_dict().items()}
    original=model.save_pretrained
    def small_shards(path, **kwargs):
        assert kwargs==dict(max_shard_size='5GB',save_original_format=False)
        kwargs['max_shard_size']='1KB'
        return original(path,**kwargs)
    model.save_pretrained=small_shards
    save,_=production_save()
    with patch('transformers.modeling_utils.revert_weight_conversion',side_effect=AssertionError('whole-model conversion called')):
        save(model,SimpleNamespace(save_pretrained=Mock()),str(tmp_path),dtype,False)
    index=json.loads((tmp_path/'model.safetensors.index.json').read_text())
    assert set(index['weight_map'])==set(expected)
    assert len(set(index['weight_map'].values()))>1
    reloaded=type(model).from_pretrained(tmp_path,dtype=getattr(torch,dtype)).eval()
    assert expected.keys()==reloaded.state_dict().keys()
    assert all(torch.equal(t,reloaded.state_dict()[k]) for k,t in expected.items())
