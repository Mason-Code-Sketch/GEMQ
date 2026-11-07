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
                num_key_value_heads=2,max_position_embeddings=2048,bos_token_id=1,eos_token_id=1,pad_token_id=1)
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
    tok=PreTrainedTokenizerFast(tokenizer_object=tok,unk_token='[UNK]',eos_token='[EOS]',pad_token='[EOS]',model_max_length=2048)
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
        result=protocol.evaluate(model,tok,'gsm8k_5shot',batch_size=1,limit=1)
    assert result['protocol']['num_fewshot']==5
    assert result['protocol']['status']=='smoke'
    assert result['summary']['sample_counts']=={'gsm8k':1}
    assert result['harness']['configs']['gsm8k']['generation_kwargs']['max_gen_toks']==512


def test_real_harness_mmlu_five_shot():
    model,tok=tiny('qwen3_moe')
    with patch('datasets.load_dataset',side_effect=dataset):
        result=protocol.evaluate(model,tok,'mmlu_5shot',batch_size=1,limit=1)
    assert len(result['summary']['subjects'])==57
    assert set(result['harness']['n-shot'].values())=={5}
    assert result['protocol']['status']=='smoke'


@pytest.mark.parametrize('profile', ['mmlu_5shot', 'gsm8k_5shot'])
def test_real_harness_auto_batch(profile):
    model, tok = tiny('qwen3_moe')
    with patch('datasets.load_dataset', side_effect=dataset):
        result = protocol.evaluate(model, tok, profile, batch_size='auto', max_batch_size=2, limit=2)
    assert result['protocol']['batch_size'] == 'auto'
    assert result['protocol']['max_batch_size'] == 2
    assert result['protocol']['status'] == 'smoke'
    assert set(result['summary']['sample_counts'].values()) == {2}
    if profile == 'gsm8k_5shot':
        probe = result['protocol']['generation_batch_probes'][0]
        assert probe['max_input_tokens'] > len(tok.encode('Question: Question 1\nAnswer:', add_special_tokens=False))
        assert probe['probe_tokens'] == probe['max_input_tokens'] + 512
        assert probe['model_max_length'] == 2048
        assert probe['selected_batch_size'] == 2
    else:
        assert result['protocol']['generation_batch_probes'] == []



def test_gsm8k_auto_probe_preserves_actual_generated_answers():
    from lm_eval.models.huggingface import HFLM
    from types import SimpleNamespace
    from gemq.generation_batch import install_generation_batch_probe
    model, tok = tiny('qwen3_moe')
    requests = [SimpleNamespace(args=(text, {'max_gen_toks':8, 'do_sample':False, 'until':['Question:']}))
                for text in ['Question: 1 Answer:', 'Question: 1 2 1 Answer:']]
    fixed = HFLM(pretrained=model, tokenizer=tok, backend='causal', batch_size=1,
                 add_bos_token=False, max_length=512)
    expected = fixed.generate_until(requests, disable_tqdm=True)
    auto = HFLM(pretrained=model, tokenizer=tok, backend='causal', batch_size='auto',
                max_batch_size=2, add_bos_token=False, max_length=512)
    records = install_generation_batch_probe(auto)
    with patch.object(auto, '_model_call', wraps=auto._model_call) as forward:
        actual = auto.generate_until(requests, disable_tqdm=True)
    assert actual == expected
    length = max(len(auto.tok_encode(r.args[0])) for r in requests) + 8
    assert records[0]['probe_tokens'] == length
    assert records[0]['selected_batch_size'] == 2
    assert forward.call_count == 5
    assert all(call.args[0].shape == (2, length) for call in forward.call_args_list)
    assert auto.max_length == 512
