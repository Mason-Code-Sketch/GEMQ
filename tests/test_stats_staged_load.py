from types import SimpleNamespace
from unittest.mock import Mock,patch
import pytest
from gemq.utils.hf_loading import load_stats_model

@pytest.mark.parametrize('integrated,count,mode,budget,staged',[
    (True,1,'layer_grads',None,True),
    (False,1,'layer_grads',None,False),
    (True,2,'layer_grads',None,False),
    (True,1,'layer_re',None,False),
    (True,1,'layer_grads',24,False),
])
def test_loading_only_changes_transient_integrated_placement(integrated,count,mode,budget,staged):
    model=Mock();factory=SimpleNamespace(from_pretrained=Mock(return_value=model))
    with patch('torch.cuda.is_available',return_value=True),patch('torch.cuda.device_count',return_value=count),patch('torch.cuda.get_device_properties',return_value=SimpleNamespace(is_integrated=integrated)),patch('torch.cuda.empty_cache'):
        result=load_stats_model(factory,'checkpoint',mode=mode,gpu_weight_budget_gib=budget,torch_dtype='float16',attn_implementation='eager',trust_remote_code=True)
    assert result is model
    args=factory.from_pretrained.call_args.kwargs
    assert args['device_map']==('cpu' if staged or mode!='layer_grads' else 'auto')
    assert args['torch_dtype']=='float16' and args['attn_implementation']=='eager' and args['trust_remote_code']
    if staged:
        model.to.assert_called_once_with('cuda:0')
        assert model.hf_device_map=={'':'cuda:0'}
    else:model.to.assert_not_called()
