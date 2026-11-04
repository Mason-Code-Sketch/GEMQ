"""CPU interface integration: actual MoE forwards and actual harness generation."""
from unittest.mock import patch
import pytest
import torch
from datasets import Dataset, DatasetDict
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast
import transformers
from gemq import supplemental as protocol


def tiny(kind):
    shared=dict(vocab_size=16,hidden_size=16,num_hidden_layers=1,num_attention_heads=2,
                num_key_value_heads=2,max_position_embeddings=1024,bos_token_id=1,eos_token_id=1,pad_token_id=1)
    if kind=='deepseek_v2':
        cfg=transformers.DeepseekV2Config(**shared,intermediate_size=32,moe_intermediate_size=16,
            n_routed_experts=4,n_shared_experts=1,num_experts_per_tok=2,q_lora_rank=8,
            kv_lora_rank=4,qk_nope_head_dim=4,qk_rope_head_dim=4,v_head_dim=4,first_k_dense_replace=0)
    elif kind=='qwen2_moe':
        cfg=transformers.Qwen2MoeConfig(**shared,intermediate_size=32,moe_intermediate_size=16,
            shared_expert_intermediate_size=16,num_experts=4,num_experts_per_tok=2)
    elif kind=='qwen3_moe':
        cfg=transformers.Qwen3MoeConfig(**shared,intermediate_size=32,moe_intermediate_size=16,
            num_experts=4,num_experts_per_tok=2,head_dim=8)
    else:
        cfg=transformers.MixtralConfig(**shared,intermediate_size=16,num_local_experts=4,num_experts_per_tok=2,head_dim=8)
    cfg._attn_implementation='eager';cfg._experts_implementation='eager'
    torch.manual_seed(42);torch.set_num_threads(1)
    model=AutoModelForCausalLM.from_config(cfg).float().eval()
    tok=Tokenizer(models.WordLevel({'[UNK]':0,'[EOS]':1,'Question':2,'Answer':3,':':4,'A':5,'B':6,'C':7,'D':8,'1':9,'2':10,'####':11},unk_token='[UNK]'))
    tok.pre_tokenizer=pre_tokenizers.Whitespace()
    tok=PreTrainedTokenizerFast(tokenizer_object=tok,unk_token='[UNK]',eos_token='[EOS]',pad_token='[EOS]',model_max_length=1024)
    return model,tok


def dataset(*args,**kwargs):
    name=args[0] if args else kwargs['path']
    if name=='cais/mmlu':
        row={'question':'Question 1','choices':['1','2','1','2'],'answer':0}
    elif name=='openai/gsm8k':row={'question':'Question 1','answer':'1 #### 1'}
    else:raise AssertionError(args)
    return DatasetDict({s:Dataset.from_list([row]*5) for s in ['train','test','dev','validation']})


@pytest.mark.parametrize('kind',['deepseek_v2','qwen2_moe','qwen3_moe','mixtral'])
def test_real_harness_gsm8k_generation(kind):
    model,tok=tiny(kind)
    # Real harness, YAML templates, extraction filters, model.generate and forward.
    # Synthetic datasets avoid downloading benchmark data or measuring model quality.
    with patch('datasets.load_dataset',side_effect=dataset):
        result=protocol.evaluate(model,tok,'gsm8k_0shot',limit=1)
    assert result['protocol']['num_fewshot']==0
    assert result['protocol']['status']=='smoke'
    assert result['summary']['sample_counts']=={'gsm8k':1}
    assert result['harness']['configs']['gsm8k']['generation_kwargs']['max_gen_toks']==256


def test_real_harness_mmlu_five_shot():
    model,tok=tiny('qwen3_moe')
    with patch('datasets.load_dataset',side_effect=dataset):
        result=protocol.evaluate(model,tok,'mmlu_5shot',limit=1)
    assert len(result['summary']['subjects'])==57
    assert set(result['harness']['n-shot'].values())=={5}
    assert result['protocol']['status']=='smoke'
