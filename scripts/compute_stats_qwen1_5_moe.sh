#!/bin/bash
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

# Model settings
model_name="Qwen/Qwen1.5-MoE-A2.7B"
model="../../models/Qwen1.5-MoE-A2.7B"
model_str=""

# Dataset settings
dataset="c4"
dataset_root="../../datasets"
nsamples=128
seqlen=2048
seed=0
resource_output="cache/${model_name}/resources/${dataset}-N${nsamples}-L${seqlen}-Seed${seed}.json"

# Step 1: layer-output gradients
layer_grads_path="cache/${model_name}/LayerGrads_${dataset}-N${nsamples}-L${seqlen}-Seed${seed}${model_str}.pt"
python -m gemq.compute_model_stats \
    --mode "layer_grads" \
    --model "${model}" \
    --model_name "${model_name}" \
    --calib_dataset "${dataset}" \
    --dataset_root "${dataset_root}" \
    --seed "${seed}" \
    --nsamples "${nsamples}" \
    --seqlen "${seqlen}" \
    --layer_grads_path "${layer_grads_path}" \
    --resource_output "${resource_output}"

# Step 2: weighted layer reconstruction errors
wbits="1,2,3"
layer_re_path="cache/${model_name}/LayerRE_${dataset}-N${nsamples}-L${seqlen}-Seed${seed}_B${wbits}${model_str}_faster.pkl"
python -m gemq.compute_model_stats \
    --mode "layer_re" \
    --model "${model}" \
    --model_name "${model_name}" \
    --calib_dataset "${dataset}" \
    --dataset_root "${dataset_root}" \
    --seed "${seed}" \
    --nsamples "${nsamples}" \
    --seqlen "${seqlen}" \
    --wbits "${wbits}" \
    --layer_grads_path "${layer_grads_path}" \
    --layer_re_path "${layer_re_path}" \
    --forward_batch_size 32 \
    --resource_output "${resource_output}"
