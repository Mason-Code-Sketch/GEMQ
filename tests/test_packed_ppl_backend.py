"""Packed PPL integration must preserve the saved HQQ checkpoint format."""
from contextlib import contextmanager
from types import ModuleType, SimpleNamespace
import sys

import pytest
import torch
from gemq import quantize


@pytest.mark.parametrize('requested,kind,dtype,expected', [
    ('auto', 'qwen3_moe', torch.float16, 'triton'),
    ('auto', 'qwen3_moe', torch.bfloat16, 'pytorch'),
    ('auto', 'mixtral', torch.float16, 'pytorch'),
    ('pytorch', 'qwen3_moe', torch.float16, 'pytorch'),
    ('triton', 'qwen3_moe', torch.float16, 'triton'),
])
def test_save_order_and_ppl_backend(monkeypatch, requested, kind, dtype, expected):
    events = []
    model = SimpleNamespace(config=SimpleNamespace(model_type=kind),
        parameters=lambda: iter([SimpleNamespace(dtype=dtype, is_cuda=True)]), layout='hqq')
    args = SimpleNamespace(packed_ppl_backend=requested, eval_real_quant=True,
        save_path='checkpoint', save_dtype='float16', real_quant=True,
        model_name='Qwen/Qwen3-30B-A3B', dataset_root='datasets')
    monkeypatch.setattr(quantize, 'release_unused_memory', lambda: None)
    def save(m, *args):
        assert m.layout == 'hqq'
        events.append('save')
    def prepare(m, *args, **kwargs):
        assert events == ['save']
        m.layout = 'triton'
        events.append('prepare')
    def evaluate(m, tokenizer, datasets, name, **kwargs):
        assert datasets == ['wikitext2', 'c4'] and kwargs['offload'] is False
        assert m.layout == ('triton' if expected == 'triton' else 'hqq')
        events.append('ppl')
    module = ModuleType('gemq.inference.patch')
    module.prepare_for_inference = prepare
    monkeypatch.setitem(sys.modules, 'gemq.inference.patch', module)
    monkeypatch.setattr(quantize, 'save_quantized_model', save)
    monkeypatch.setattr(quantize, 'evaluate_perplexity', evaluate)
    @contextmanager
    def component(_):
        yield
    quantize.finalize_packed_checkpoint(model, None, args, SimpleNamespace(component=component))
    assert events == (['save', 'prepare', 'ppl'] if expected == 'triton' else ['ppl', 'save'])


def test_explicit_unsupported_backend_rejected_before_save(monkeypatch):
    model = SimpleNamespace(config=SimpleNamespace(model_type='mixtral'),
        parameters=lambda: iter([SimpleNamespace(dtype=torch.float16, is_cuda=True)]))
    args = SimpleNamespace(packed_ppl_backend='triton', eval_real_quant=True)
    with pytest.raises(ValueError, match='Qwen3 FP16'):
        quantize.finalize_packed_checkpoint(model, None, args, None)
