import importlib.util
from pathlib import Path
import pytest
import torch

def load(path, name):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

@pytest.mark.parametrize('cls',['GPTQWeightQuantizer','MCMoeGPTQWeightQuantizer'])
@pytest.mark.parametrize('bits',[1,2,3,4])
@pytest.mark.parametrize('mcmoe',[False,True])
@pytest.mark.parametrize('groupsize',[128,-1])
def test_quantizer_outputs_exact(cls,bits,mcmoe,groupsize):
    root=Path(__file__).resolve().parents[1]
    baseline=load(root/'tests/fixtures/gptq_before_speed.py','baseline_speed')
    candidate=load(root/'gemq/quantizers/gptq.py','candidate_speed')
    torch.manual_seed(0)
    weight=torch.randn(8,256);weight[0]=0;weight[:,0]=0
    inp=torch.randn(300,256)
    if cls=='MCMoeGPTQWeightQuantizer' and groupsize==-1:
        for module in [baseline,candidate]:
            with pytest.raises(ValueError,match='groupsize > 0'):
                getattr(module,cls)(weight,nbits=bits,groupsize=groupsize,mcmoe_rules=mcmoe)
        return
    outputs=[]
    for module in [baseline,candidate]:
        q=getattr(module,cls)(weight,nbits=bits,groupsize=groupsize,mcmoe_rules=mcmoe)
        q.add_batch(inp);outputs.append(q.quantize())
    assert all(torch.equal(a,b) for a,b in zip(*outputs))
