#!/bin/bash
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

# NOTE: DeepSeek-V2-Lite has two implementations: HuggingFace Transformers' built-in one
# and the official one shipped with the weights (trust_remote_code).
#
# The built-in one used to score ~15% worse (10.80 vs 9.39 wikitext2 ppl). Cause: it omits
# the YaRN mscale that the official code folds into the attention softmax scale. Both
# entry points now call gemq.utils.hf_loading.align_deepseek_softmax_scale, which restores
# it, so the two agree to within ~0.1% and either flag setting is fine here.
#
# The official implementation predates `cache_position` and the transformers 4.5x Cache
# API, so it cannot be used for generation -- see scripts/bench_generate_deepseekv2.sh.

# ===============================
#  Model settings
# ===============================
model_name="deepseek-ai/DeepSeek-V2-Lite"
model="../../models/DeepSeek-V2-Lite"
use_official_impl=true     # use official modeling code (set false to use HF Transformers implementation)

# ===============================
#  Dataset settings
# ===============================
calib_dataset="wikitext2"
dataset_root="../../datasets"
nsamples=128
seqlen=2048

# ===============================
#  Quantization settings
# ===============================
quantizer="gptq"
bpe=2.0                    # bits per expert
mixed_prec=true            # enable expert-level mixed-precision quantization (set false for uniform quantization)
bit_cfg="configs/${model_name}/GEMQ/C4-Seed0_E${bpe}_B1,2,3_c2c3.pkl"

# ===============================
#  Router fine-tuning
# ===============================
# NOTE: Router fine-tuning for DeepSeek-V2-Lite requires 2×80GB GPUs.
# Set this option to false if you do not have sufficient resources.
finetune_routers=true      # whether to finetune the routers after quantization
rft_epochs=1
rft_lr=1e-4

# ===============================
#  Evaluation settings
# ===============================
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
if [[ "${use_official_impl}" == "true" ]]; then
    model_args+=(--trust_remote_code)
fi

data_args=(--calib_dataset "$calib_dataset" --dataset_root "$dataset_root" --nsamples "$nsamples" --seqlen "$seqlen")

bpe_int=$(printf "%.0f" "$bpe")
quant_args=(--quantizer "$quantizer" --expert_wbits "$bpe_int" --groupsize 128 --mse --reproduce_mcmoe)
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

eval_args=()
if [[ "${eval_downstream}" == "true" ]]; then
    eval_args=(--eval_downstream --downstream_tasks "$downstream_tasks")
    if [[ "${use_official_impl}" == "true" ]]; then
        eval_args+=(--disable_cache)
    fi
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
    resource_output="${save_path}/resource_breakdown.json"
else
    save_path="None"
    io_args=()
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
echo " Implementation:   $( [[ "${use_official_impl}" == "true" ]] && echo 'Official (trust_remote_code)' || echo 'Transformers')"
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
