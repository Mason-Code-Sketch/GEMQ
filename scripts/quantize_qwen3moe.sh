#!/bin/bash
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

# ===============================
#  Model settings
# ===============================
model_name="Qwen/Qwen3-30B-A3B"
model="Qwen/Qwen3-30B-A3B"

# ===============================
#  Dataset settings
# ===============================
calib_dataset="wikitext2"
nsamples=128
seqlen=2048

# ===============================
#  Quantization settings
# ===============================
quantizer="gptq"
bpe=2.5                    # bits per expert
mixed_prec=true            # enable expert-level mixed-precision quantization (set false for uniform quantization)
bit_cfg="configs/${model_name}/GEMQ/C4-Seed0_E${bpe}_B1,2,3_c2c3.pkl"

# ===============================
#  Router fine-tuning
# ===============================
# NOTE: Router fine-tuning for Mixtral-8x7B requires 3×80GB GPUs.
# Set this option to false if you do not have sufficient resources.
finetune_routers=false      # whether to finetune the routers after quantization
rft_epochs=1
rft_lr=1e-4

# ===============================
#  Evaluation settings
# ===============================
# NOTE: 2x 80G GPUs are required for downstream eval
eval_downstream=false      # whether to run downstream eval after quantization
downstream_tasks="piqa,arc_easy,arc_challenge,hellaswag,winogrande,mathqa,mmlu"

# ===============================
#  I/O settings
# ===============================
real_quant=true            # whether to pack + save INT weights (set false for pseudo quantization)
save_model=true            # whether to save the quantized model



# ===============================
#  AUTO argument construction
# ===============================
model_args=(--model "$model" --model_name "$model_name")

data_args=(--calib_dataset "$calib_dataset" --nsamples "$nsamples" --seqlen "$seqlen")

# NOTE: `--reproduce_mcmoe` may cause NaNs. Disable it or use BF16 instead.
bpe_int=$(printf "%.0f" "$bpe")
quant_args=(--quantizer "$quantizer" --expert_wbits "$bpe_int" --groupsize 128 --mse)
if [[ "${mixed_prec}" == "true" ]]; then
    qtype="$(basename "$(dirname "$bit_cfg")")"
    quant_args+=(--mixed --bit_cfg "$bit_cfg")
else
    qtype="Uniform"
fi

rft_tag=""
if [[ "${finetune_routers}" == "true" ]]; then
    rft_tag="_RFT"
    quant_args+=(--finetune_routers --rft_epochs "$rft_epochs" --rft_lr "$rft_lr")
fi

if [[ "${eval_downstream}" == "true" ]]; then
    quant_args+=(--eval_downstream --downstream_tasks "$downstream_tasks")
fi

fname="${bit_cfg##*/}"
alloc_prefix="${fname%%_*}"
prefix="${alloc_prefix}-WT2"
if [[ "${save_model}" == "true" ]]; then
    if [[ "${real_quant}" == "true" ]]; then
        save_path="results/real_quant_models/${model_name}/${qtype}/${prefix}_A4-G16-D4-E${bpe}${rft_tag}"
        io_args=(--real_quant --save_path "$save_path")
    else
        save_path="results/fake_quant_models/${model_name}/${qtype}/${prefix}_A4-G16-D4-E${bpe}${rft_tag}"
        io_args=(--save_path "$save_path")
    fi
else
    save_path="None"
    io_args=()
fi

if [[ "${save_model}" == "true" ]]; then
    resource_output="${save_path}/resource_breakdown.json"
else
    resource_output="results/resource-records/${model_name}/${qtype}/${prefix}_A4-G16-D4-E${bpe}${rft_tag}.json"
fi
io_args+=(--resource_output "$resource_output")


# ===============================
#  Run
# ===============================
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
echo "----------------------------------------------"
echo ">>> Running quantization ..."
echo "=============================================="

python -m gemq.quantize \
    "${model_args[@]}" \
    "${data_args[@]}" \
    "${quant_args[@]}" \
    "${io_args[@]}"

if [[ "${mixed_prec}" == "true" ]]; then
    stats_resource="cache/${model_name}/LayerRE_c4-N128-L2048-Seed0_B1,2,3_faster.resource.json"
    allocation_resource="${bit_cfg%.pkl}.resource.json"
    python - "${resource_output}" "${stats_resource}" "${allocation_resource}" <<'PY'
import sys

from gemq.resource_ledger import merge_ledgers

output_path, *upstream_paths = sys.argv[1:]
merge_ledgers(output_path, [output_path, *upstream_paths])
PY
fi
