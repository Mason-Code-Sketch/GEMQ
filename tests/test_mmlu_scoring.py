"""Compare the optimized path with native HFLM on real, small MoE models."""
import pytest
import torch
from unittest.mock import Mock
from transformers import (Qwen2MoeConfig, Qwen2MoeForCausalLM,
    Qwen3MoeConfig, Qwen3MoeForCausalLM, MixtralConfig, MixtralForCausalLM)
from lm_eval.models.huggingface import HFLM
from gemq.mmlu_scoring import compact_mmlu_scoring


def fixture_model(kind):
    torch.manual_seed(17)
    config_cls, model_cls = {'qwen2': (Qwen2MoeConfig, Qwen2MoeForCausalLM),
        'qwen3': (Qwen3MoeConfig, Qwen3MoeForCausalLM),
        'mixtral': (MixtralConfig, MixtralForCausalLM)}[kind]
    config = config_cls(vocab_size=64, hidden_size=32, intermediate_size=48,
        moe_intermediate_size=16, shared_expert_intermediate_size=16,
        num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2,
        head_dim=8, num_experts=4, num_local_experts=4, num_experts_per_tok=2,
        max_position_embeddings=128, use_cache=False)
    config._attn_implementation='eager'
    return model_cls(config).eval()


def harness(model, batch, cache):
    h=HFLM.__new__(HFLM); h._model=model; h.backend='causal'; h._device=torch.device('cpu')
    h._rank=0; h._world_size=1; h._max_length=12; h.batch_size_per_gpu=batch
    h.logits_cache=cache; h.softmax_dtype=torch.float32; h.mixed_precision_dtype=None
    h.cache_hook=Mock(); return h


@pytest.mark.parametrize('kind',['qwen2','qwen3','mixtral'])
@pytest.mark.parametrize('batch',[1,3])
@pytest.mark.parametrize('cache',[False,True])
def test_native_scores_order_grouping_padding_truncation(kind,batch,cache):
    h=harness(fixture_model(kind),batch,cache)
    requests=[((str(n),str(t)),list(range(1,n+1)),[t])
              for n in [5,17,8,5] for t in [21,22,23,24]]
    expected=h._loglikelihood_tokens(requests,disable_tqdm=True)
    projected=[]
    hook=h.model.lm_head.register_forward_pre_hook(lambda m,args: projected.append(args[0].shape[1]))
    try:
        with compact_mmlu_scoring(h) as stats:
            actual=h._loglikelihood_tokens(requests,disable_tqdm=True)
        assert stats['optimized_requests']==len(requests)
        assert max(projected)<=batch
        assert [x[1] for x in actual]==[x[1] for x in expected]
        assert [x[0] for x in actual]==pytest.approx([x[0] for x in expected],abs=1e-6)
        for offset in range(0,len(requests),4):
            assert max(range(4),key=lambda i:actual[offset+i][0])==max(range(4),key=lambda i:expected[offset+i][0])
        assert '_loglikelihood_tokens' not in h.__dict__
    finally: hook.remove()


def test_multitoken_falls_back_and_exception_restores():
    h=harness(fixture_model('qwen2'),2,True)
    requests=[(('ctx','answer'),[1,2,3],[4,5])]
    expected=h._loglikelihood_tokens(requests,disable_tqdm=True)
    with pytest.raises(RuntimeError,match='deliberate'):
        with compact_mmlu_scoring(h) as stats:
            assert h._loglikelihood_tokens(requests,disable_tqdm=True)==expected
            assert stats['fallback_calls']==1
            raise RuntimeError('deliberate')
    assert '_loglikelihood_tokens' not in h.__dict__


def test_unsupported_keeps_native():
    h=harness(fixture_model('qwen2'),2,True);h.model.config.model_type='deepseek_v2'
    native=h._loglikelihood_tokens
    with compact_mmlu_scoring(h) as stats:
        assert not stats['enabled'];assert h._loglikelihood_tokens==native


def test_qwen_cache_off_preserves_logits_and_eliminates_returned_cache():
    model=fixture_model('qwen2');inputs=torch.tensor([[1,2,3,4]])
    with torch.no_grad():
        cached=model(inputs,use_cache=True)
        uncached=model(inputs,use_cache=False)
    assert cached.past_key_values is not None
    assert uncached.past_key_values is None
    torch.testing.assert_close(cached.logits,uncached.logits,rtol=0,atol=1e-6)


def test_auto_scheduler_uses_cap_and_restores_previous_override():
    h=harness(fixture_model('qwen2'),'auto',True)
    h.max_batch_size=2;h.batch_schedule=1;h.batch_sizes={0:99}
    h._detect_batch_size=Mock(return_value=2)
    previous=h._loglikelihood_tokens
    h._loglikelihood_tokens=previous
    requests=[((str(n),'A'),list(range(1,n+1)),[21]) for n in [5,8,7]]
    with compact_mmlu_scoring(h):
        results=h._loglikelihood_tokens(requests,disable_tqdm=True)
    assert len(results)==3
    h._detect_batch_size.assert_called_once()
    assert h.batch_sizes=={0:2}
    assert h._loglikelihood_tokens==previous
