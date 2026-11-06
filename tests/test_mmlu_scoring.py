"""Compare the optimized path with native HFLM on real, small MoE models."""
import pytest
import torch
from unittest.mock import Mock
from transformers import (Qwen2MoeConfig, Qwen2MoeForCausalLM,
    Qwen3MoeConfig, Qwen3MoeForCausalLM, MixtralConfig, MixtralForCausalLM, DeepseekV2Config, DeepseekV2ForCausalLM)
from lm_eval.models.huggingface import HFLM
from gemq.mmlu_scoring import compact_mmlu_scoring


def fixture_model(kind):
    torch.manual_seed(17)
    if kind in {'deepseek', 'deepseek_legacy'}:
        config = DeepseekV2Config(vocab_size=64, hidden_size=32, intermediate_size=48,
            moe_intermediate_size=16, num_hidden_layers=2, num_attention_heads=4,
            num_key_value_heads=4, kv_lora_rank=8, q_lora_rank=None,
            qk_nope_head_dim=4, qk_rope_head_dim=4, v_head_dim=4,
            n_routed_experts=4, n_shared_experts=1, num_experts_per_tok=2,
            first_k_dense_replace=1, max_position_embeddings=128, use_cache=False)
        config._attn_implementation='eager'
        config._experts_implementation='eager'
        cls = DeepseekV2ForCausalLM
        if kind == 'deepseek_legacy':
            # Check the checkpoint forward contract: full hidden states enter
            # lm_head, then logits are upcast. Native decoder is a CPU fixture.
            def forward(self, input_ids, **kwargs):
                from transformers.modeling_outputs import CausalLMOutputWithPast
                outputs = self.model(input_ids=input_ids, **kwargs)
                return CausalLMOutputWithPast(logits=self.lm_head(outputs[0]).float())
            cls = type('DeepseekV2ForCausalLM', (cls,),
                       {'forward': forward, '__module__': 'transformers_modules.deepseek_test'})
        return cls(config).eval()
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


@pytest.mark.parametrize('kind',['qwen2','qwen3','mixtral','deepseek','deepseek_legacy'])
@pytest.mark.parametrize('batch',[1,3])
@pytest.mark.parametrize('cache',[False,True])
def test_native_scores_order_grouping_padding_truncation(kind,batch,cache):
    h=harness(fixture_model(kind),batch,cache)
    requests=[((str(n),str(t)),list(range(1,n+1)),[t])
              for n in [5,17,8,5] for t in [21,22,23,24]]
    expected=h._loglikelihood_tokens(requests,disable_tqdm=True)
    projected=[]
    hook=h.model.lm_head.register_forward_hook(lambda m,args,out: projected.append(out.shape[1]))
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
    h=harness(fixture_model('qwen2'),2,True);h.model.config.model_type='unsupported_model'
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
    h._detect_batch_size.assert_not_called()
    assert h.batch_sizes=={0:2}
    assert h._loglikelihood_tokens==previous


def test_legacy_head_hook_removed_on_forward_error():
    h=harness(fixture_model('deepseek_legacy'),2,True)
    before=dict(h.model.lm_head._forward_pre_hooks)
    def fail(*args):
        raise RuntimeError('head failure')
    hook=h.model.lm_head.register_forward_hook(fail)
    try:
        with compact_mmlu_scoring(h):
            with pytest.raises(RuntimeError, match='head failure'):
                h._loglikelihood_tokens([(('x','A'),[1,2,3],[21])],disable_tqdm=True)
        assert dict(h.model.lm_head._forward_pre_hooks)==before
        assert '_loglikelihood_tokens' not in h.__dict__
    finally:
        hook.remove()


@pytest.mark.parametrize('kind',['qwen3','deepseek_legacy'])
def test_compact_probe_five_forwards_halves_oom_and_restores(kind):
    h=harness(fixture_model(kind),'auto',True)
    h.max_batch_size=4
    native=h._detect_batch_size
    seen=[]
    def record(_m,args,out):
        seen.append(tuple(out.shape))
        if out.shape[0]>2:
            raise RuntimeError('CUDA out of memory.')
    hook=h.model.lm_head.register_forward_hook(record)
    try:
        with compact_mmlu_scoring(h):
            assert h._detect_batch_size([(('x','A'),list(range(1,10)),[21])])==2
        assert h._detect_batch_size==native
        assert len([shape for shape in seen if shape[0]==2])==5
        assert all(shape[1]<=shape[0] for shape in seen)
    finally:
        hook.remove()
