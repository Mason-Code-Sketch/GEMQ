"""Protocol tests shared by all six independently importable branches."""
import copy
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock
import sys
import pytest

from gemq import supplemental as protocol


def raw_result(profile, count=None):
    if profile == 'gsm8k_0shot':
        return {'results': {'gsm8k': {'exact_match,strict-match': .25, 'exact_match,flexible-extract': .5}},
                'n-shot': {'gsm8k': 0}, 'n-samples': {'gsm8k': {'effective': count or 1319}},
                'configs': {'gsm8k': {'num_fewshot': 0, 'generation_kwargs': {'until': ['Question:']}}}}
    names = ['mmlu_'+str(i) for i in range(57)]
    counts = {n: (246 if i < 56 else 266) for i,n in enumerate(names)}
    if count: counts = {n: count for n in names}
    return {'results': {n: {'acc,none': .5} for n in names}, 'groups': {'mmlu': {'acc,none': .5}},
            'n-shot': {n: 5 for n in names}, 'n-samples': {n: {'effective': c} for n,c in counts.items()},
            'configs': {n: {'num_fewshot': 5} for n in names}}


@pytest.mark.parametrize('profile', list(protocol.PROFILES))
def test_full_summary(profile):
    s=protocol.summarize(raw_result(profile), profile)
    assert 'seven_task_average' not in s
    assert sum(s['sample_counts'].values()) == (14042 if profile == 'mmlu_5shot' else 1319)


@pytest.mark.parametrize('profile', list(protocol.PROFILES))
@pytest.mark.parametrize('damage', ['shots','samples','nan'])
def test_invalid_results_fail(profile, damage):
    raw=raw_result(profile); name=next(iter(raw['n-shot']))
    if damage=='shots':raw['n-shot'][name]+=1
    elif damage=='samples':raw['n-samples'][name]['effective']-=1
    else:raw['results'][name][next(iter(raw['results'][name]))]=float('nan')
    with pytest.raises(ValueError):protocol.summarize(raw,profile)


def test_mmlu_must_be_sample_weighted():
    r=raw_result('mmlu_5shot'); r['results']['mmlu_56']['acc,none']=1
    with pytest.raises(ValueError,match='aggregate'):protocol.summarize(r,'mmlu_5shot')
    r['groups']['mmlu']['acc,none']=(.5*13776+266)/14042
    assert protocol.summarize(r,'mmlu_5shot')['mmlu']['value']==r['groups']['mmlu']['acc,none']


@pytest.mark.parametrize('profile', list(protocol.PROFILES))
def test_smoke_is_distinct(profile):
    r=raw_result(profile,1)
    with pytest.raises(ValueError):protocol.summarize(r,profile)
    protocol.summarize(r,profile,limit=1)


@pytest.mark.parametrize('profile', list(protocol.PROFILES))
@pytest.mark.parametrize('model_type', ['deepseek_v2', 'qwen2_moe', 'qwen3_moe', 'mixtral'])
@pytest.mark.parametrize('explicit', [None, 5])
def test_real_interface_arguments(monkeypatch,profile,model_type,explicit):
    raw=raw_result(profile); call=Mock(return_value=raw); hf=Mock()
    pkg=ModuleType('lm_eval');pkg.evaluator=SimpleNamespace(simple_evaluate=call)
    mod=ModuleType('lm_eval.models.huggingface');mod.HFLM=hf
    monkeypatch.setitem(sys.modules,'lm_eval',pkg);monkeypatch.setitem(sys.modules,'lm_eval.models.huggingface',mod)
    monkeypatch.setattr(protocol,'version',lambda _: '0.4.13')
    model=Mock(); model.config=SimpleNamespace(model_type=model_type,use_cache=True)
    def evaluate_callback(**kwargs):
        expected_cache = not (model_type == 'deepseek_v2' or
                              (model_type == 'qwen2_moe' and profile == 'mmlu_5shot'))
        assert model.config.use_cache is expected_cache
        return raw
    call.side_effect = evaluate_callback
    result=protocol.evaluate(model,'tokenizer',profile,max_batch_size=explicit)
    args=call.call_args.kwargs
    assert args['tasks']==[protocol.PROFILES[profile][0]]
    assert args['num_fewshot']==protocol.PROFILES[profile][1]
    assert args['apply_chat_template'] is False and args['fewshot_random_seed']==0
    assert hf.call_args.kwargs['parallelize'] is False
    assert hf.call_args.kwargs['add_bos_token'] is False
    assert hf.call_args.kwargs['batch_size']=='auto'
    caps = {
        'qwen3_moe': {'mmlu_5shot': 60, 'gsm8k_0shot': 96},
        'mixtral': {'mmlu_5shot': 13, 'gsm8k_0shot': 48},
        'deepseek_v2': {'mmlu_5shot': 3, 'gsm8k_0shot': 16},
        'qwen2_moe': {'mmlu_5shot': 5, 'gsm8k_0shot': 24},
    }.get(model_type, {'mmlu_5shot': 17, 'gsm8k_0shot': 96})
    expected = explicit if explicit is not None else caps[profile]
    assert hf.call_args.kwargs['max_batch_size'] == expected
    assert result['protocol']['max_batch_size'] == hf.call_args.kwargs['max_batch_size']
    assert model.config.use_cache is True
    if profile=='gsm8k_0shot':
        assert args['gen_kwargs']=={'do_sample':False,'temperature':0.0,'max_gen_toks':256}
    else:assert 'gen_kwargs' not in args
    assert result['protocol']['status']=='complete'
    assert len(result['protocol']['task_configs_sha256'])==64
    call.side_effect=RuntimeError('generation failed')
    with pytest.raises(RuntimeError):protocol.evaluate(model,'tokenizer',profile)
    assert model.config.use_cache is True


def test_mmlu_category_groups_are_not_subjects():
    raw=raw_result('mmlu_5shot')
    raw['results']['mmlu_stem']={'acc,none': .5}
    raw['n-shot']['mmlu_stem']=5
    assert len(protocol.summarize(raw,'mmlu_5shot')['subjects'])==57
