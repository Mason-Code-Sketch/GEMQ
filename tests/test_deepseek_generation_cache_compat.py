from types import SimpleNamespace
import pytest
import torch
from transformers.cache_utils import DynamicCache
from gemq.utils.hf_loading import install_deepseek_generation_cache_compat

class Legacy:
    __module__ = 'transformers_modules.test_deepseek'
    def __init__(self):
        self.config = SimpleNamespace(model_type='deepseek_v2')
    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, **kwargs):
        c = past_key_values
        if c is None:
            return input_ids
        assert c.get_max_length() is None
        return input_ids[:, c.seen_tokens:], c.get_usable_length(1, 0), c.get_usable_length(1, 1)

def test_dynamic_cache_prefill_decode_and_layer_lengths():
    m=Legacy();assert install_deepseek_generation_cache_compat(m)
    c=DynamicCache();ids=torch.arange(5)[None]
    assert m.prepare_inputs_for_generation(ids, past_key_values=c)[0].shape==(1,5)
    for i in range(2):
        k=torch.randn(1,2,5,3);c.update(k,k,i)
    y,n0,n1=m.prepare_inputs_for_generation(torch.arange(6)[None],past_key_values=c)
    assert y.tolist()==[[5]] and (n0,n1)==(5,5)
    k=torch.randn(1,2,1,3);c.update(k,k,0)
    assert c.get_usable_length(1,0)==6 and c.get_usable_length(1,1)==5
    c.update(k,k,1)
    assert m.prepare_inputs_for_generation(torch.arange(7)[None],past_key_values=c)[0].tolist()==[[6]]
    assert not install_deepseek_generation_cache_compat(m)

def test_other_models_untouched():
    m=Legacy();m.config.model_type='qwen2_moe';old=m.prepare_inputs_for_generation
    assert not install_deepseek_generation_cache_compat(m)
    assert m.prepare_inputs_for_generation==old

def test_no_cache_and_offload_rejection():
    m=Legacy();install_deepseek_generation_cache_compat(m);ids=torch.ones((1,2),dtype=torch.long)
    assert m.prepare_inputs_for_generation(ids) is ids
    c=DynamicCache();c.offloading=True
    with pytest.raises(ValueError,match='resident'):
        m.prepare_inputs_for_generation(ids,past_key_values=c)


def test_compat_does_not_keep_generation_cache_alive():
    import weakref
    m=Legacy();install_deepseek_generation_cache_compat(m)
    c=DynamicCache();weak=weakref.ref(c)
    m.prepare_inputs_for_generation(torch.ones((1,2),dtype=torch.long),past_key_values=c)
    del c
    assert weak() is None
