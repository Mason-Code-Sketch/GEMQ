"""One real-size Mixtral layer, GPU serialization memory comparison."""
import gc,json,pathlib,subprocess,tempfile,time
import torch
from transformers import MixtralConfig,MixtralForCausalLM

assert pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip()=='9927175a-6e23-46f8-baa3-b12916eefe82'
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
torch.manual_seed(0)
c=MixtralConfig(vocab_size=32,hidden_size=4096,intermediate_size=14336,num_hidden_layers=1,
                num_attention_heads=32,num_key_value_heads=8,num_local_experts=8,num_experts_per_tok=2)
model=MixtralForCausalLM(c).half().cuda().eval()
results=[]
for original in [True,False]:
 gc.collect();torch.cuda.empty_cache();torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
 before=torch.cuda.memory_allocated();start=time.perf_counter()
 with tempfile.TemporaryDirectory() as directory:
  model.save_pretrained(directory,max_shard_size='5GB',save_original_format=original)
  torch.cuda.synchronize()
  record=dict(save_original_format=original,seconds=time.perf_counter()-start,
              base_gpu_bytes=before,peak_gpu_bytes=torch.cuda.max_memory_allocated(),
              extra_gpu_bytes=torch.cuda.max_memory_allocated()-before)
  restored=MixtralForCausalLM.from_pretrained(directory,dtype=torch.float16)
  assert model.state_dict().keys()==restored.state_dict().keys()
  assert all(torch.equal(t.detach().cpu(),restored.state_dict()[k]) for k,t in model.state_dict().items())
  record['all_weights_exact']=True
  del restored
 results.append(record)
 print(json.dumps(record),flush=True)
out=dict(status='complete',scope='Synthetic weights, real Mixtral dimensions, one layer, original/native layout round trips',results=results)
pathlib.Path(__file__).with_name('gpu-save-results.json').write_text(json.dumps(out,indent=2))
