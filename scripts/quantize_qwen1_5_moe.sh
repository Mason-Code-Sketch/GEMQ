#!/bin/bash
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

# Model settings
model_name="Qwen/Qwen1.5-MoE-A2.7B"
model="../../models/Qwen1.5-MoE-A2.7B"

# Dataset settings
calib_dataset="wikitext2"
dataset_root="../../datasets"
nsamples=128
seqlen=2048

# Quantization settings
quantizer="gptq"
bpe=2.0
mixed_prec=true
bit_cfg="configs/${model_name}/GEMQ/C4-Seed0_E${bpe}_B1,2,3_c2c3.pkl"

# Router fine-tuning
finetune_routers=true
rft_epochs=1
rft_lr=1e-4

# Evaluation settings
eval_downstream=false
downstream_tasks="piqa,arc_easy,arc_challenge,hellaswag,winogrande,mathqa,mmlu"

# Qwen1.5 stores routed experts in fused parameter tensors. GEMQ can save its
# pseudo-quantized checkpoint, while HQQ packing is unavailable for this layout.
real_quant=false
save_model=true

model_args=(--model "$model" --model_name "$model_name")
data_args=(--calib_dataset "$calib_dataset" --dataset_root "$dataset_root" --nsamples "$nsamples" --seqlen "$seqlen")
bpe_int=$(printf "%.0f" "$bpe")
quant_args=(--quantizer "$quantizer" --expert_wbits "$bpe_int" --groupsize 128 --mse --reproduce_mcmoe)
if [[ "$mixed_prec" == "true" ]]; then
    qtype="$(basename "$(dirname "$bit_cfg")")"
    quant_args+=(--mixed --bit_cfg "$bit_cfg")
else
    qtype="Uniform"
fi

rft_tag=""
if [[ "$finetune_routers" == "true" ]]; then
    rft_tag="_RFT"
    quant_args+=(--finetune_routers --rft_epochs "$rft_epochs" --rft_lr "$rft_lr")
fi

eval_args=()
if [[ "$eval_downstream" == "true" ]]; then
    eval_args=(--eval_downstream --downstream_tasks "$downstream_tasks")
fi

fname="${bit_cfg##*/}"
alloc_prefix="${fname%%_*}"
prefix="${alloc_prefix}-WT2"
if [[ "$save_model" == "true" ]]; then
    save_path="results/fake_quant_models/${model_name}/${qtype}/${prefix}_A4-G16-D4-E${bpe}${rft_tag}"
    io_args=(--save_path "$save_path")
    resource_output="${save_path}/resource_breakdown.json"
else
    save_path="None"
    io_args=()
    resource_output="results/resource-records/${model_name}/${qtype}/${prefix}_A4-G16-D4-E${bpe}${rft_tag}.json"
fi
io_args+=(--resource_output "$resource_output")

echo "=============================================="
echo ">>> Quantization Job Summary"
echo "----------------------------------------------"
echo " Model:            ${model_name}"
echo " Dataset:          ${calib_dataset} (nsamples=${nsamples}, seqlen=${seqlen})"
echo "----------------------------------------------"
echo " Quantizer:        ${quantizer}"
echo " Expert bits:      ${bpe} (mixed: ${mixed_prec})"
echo " Bit config:       ${bit_cfg}"
echo " Finetune routers: ${finetune_routers} (epochs=${rft_epochs}, lr=${rft_lr})"
echo " Save path:        ${save_path}"
echo " Resource record:  ${resource_output}"
echo "----------------------------------------------"
echo ">>> Running quantization ..."
echo "=============================================="

python -m gemq.quantize \
    "${model_args[@]}" \
    "${data_args[@]}" \
    "${quant_args[@]}" \
    "${eval_args[@]}" \
    "${io_args[@]}"

stats_resource="cache/${model_name}/resources/c4-N128-L2048-Seed0.json"
allocation_resource="${bit_cfg%.pkl}.resource.json"
python - "${resource_output}" "${stats_resource}" "${allocation_resource}" <<'PY'
import sys

from gemq.resource_ledger import merge_ledgers

output_path, *upstream_paths = sys.argv[1:]
merge_ledgers(output_path, [output_path, *upstream_paths])
PY
