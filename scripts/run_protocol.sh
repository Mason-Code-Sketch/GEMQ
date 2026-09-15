#!/usr/bin/env bash
set -euo pipefail

# Run the GEMQ protocol with local model and dataset directories.  All generated
# artifacts are rooted under results/protocol so source and external data paths
# remain untouched.
if [[ $# -lt 2 ]]; then
    echo "Usage: $0 <protocol-env> <stats|allocate|quantize|bootstrap|progressive> ..."
    exit 2
fi

config_file="$1"
stage="$2"
bit_budget="${3:-}"

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
source "$config_file"

if [[ -z "${MODEL_NAME:-}" || -z "${MODEL_PATH:-}" || -z "${DATASET_ROOT:-}" ]]; then
    echo "Protocol config must define MODEL_NAME, MODEL_PATH, and DATASET_ROOT."
    exit 2
fi

model_key="$(basename "$config_file" .env)"
artifact_root="results/protocol/${model_key}"
stats_root="${artifact_root}/statistics"
alloc_root="${artifact_root}/allocations"
checkpoint_root="${artifact_root}/checkpoints"
python_bin="${PYTHON_BIN:-python}"

common_model_args=(
    --model "$MODEL_PATH"
    --model_name "$MODEL_NAME"
    --model_dtype "${MODEL_DTYPE:-float16}"
    --attn_impl "${ATTN_IMPL:-eager}"
)
common_data_args=(
    --dataset_root "$DATASET_ROOT"
    --experiment_protocol "${EXPERIMENT_PROTOCOL:-vivit_ggn}"
    --seed "${SEED:-0}"
    --nsamples "${NSAMPLES:-128}"
    --seqlen "${SEQLEN:-2048}"
)
gptq_args=()
case "${GPTQ_IMPLEMENTATION:-mcmoe}" in
    mcmoe)
        gptq_args=(--reproduce_mcmoe)
        ;;
    gemq)
        ;;
    *)
        echo "GPTQ_IMPLEMENTATION must be 'mcmoe' or 'gemq'."
        exit 2
        ;;
esac

stage_paths() {
    local stage_label="$1"
    layer_grads_path="${stats_root}/${stage_label}/layer_grads_c4_seed0.pt"
    layer_re_path="${stats_root}/${stage_label}/layer_re_c4_seed0_bits123.pkl"
}

run_logged() {
    local log_name="$1"
    shift
    local log_path="${artifact_root}/logs/${log_name}.log"
    mkdir -p "$(dirname "$log_path")"
    "$@" 2>&1 | tee -a "$log_path"
}

run_stats() {
    local stage_label="$1"
    local importance_model="$2"
    stage_paths "$stage_label"
    mkdir -p "$(dirname "$layer_grads_path")"
    run_logged "${stage_label}_layer_grads" "$python_bin" -m gemq.compute_model_stats \
        --mode layer_grads \
        --model "$importance_model" \
        "${common_model_args[@]:2}" \
        "${common_data_args[@]}" \
        --calib_dataset c4 \
        --layer_grads_path "$layer_grads_path"
    run_logged "${stage_label}_layer_re" "$python_bin" -m gemq.compute_model_stats \
        --mode layer_re \
        --model "$importance_model" \
        "${common_model_args[@]:2}" \
        "${common_data_args[@]}" \
        --calib_dataset c4 \
        --wbits 1,2,3 \
        --layer_grads_path "$layer_grads_path" \
        --layer_re_path "$layer_re_path" \
        --forward_batch_size "${FORWARD_BATCH_SIZE:-1}"
}

run_allocate() {
    local stage_label="$1"
    local target_bit="$2"
    stage_paths "$stage_label"
    local allocation_path="${alloc_root}/${stage_label}/bits123_avg${target_bit}.pkl"
    mkdir -p "$(dirname "$allocation_path")"
    run_logged "${stage_label}_allocate_avg${target_bit}" "$python_bin" -m gemq.allocate_bits \
        --model_name "$MODEL_NAME" \
        --layer_re_path "$layer_re_path" \
        --bit_budget "$target_bit" \
        --bit_candidates 1,2,3 \
        --ilp_solver gemq \
        --ilp_backend "${ILP_BACKEND:-highs}" \
        --save_path "$allocation_path"
}

run_quantize() {
    local stage_label="$1"
    local target_bit="$2"
    local importance_model="$3"
    local allocation_path="${alloc_root}/${stage_label}/bits123_avg${target_bit}.pkl"
    if [[ ! -f "$allocation_path" ]]; then
        echo "Missing allocation: $allocation_path"
        exit 2
    fi
    mkdir -p "$checkpoint_root"
    quantize_args=(
        --calib_dataset wikitext2
        --quantizer gptq
        --mixed
        --bit_cfg "$allocation_path"
        --groupsize 128
        --mse
        "${gptq_args[@]}"
        --attn_wbits 4
        --dense_wbits 4
        --gate_wbits 16
        --expert_wbits 3
        --skip_pre_finetune_eval
        --save_dtype "${SAVE_DTYPE:-float16}"
        --save_path "${checkpoint_root}/avg${target_bit}"
    )
    if [[ "${FINETUNE_ROUTERS:-false}" == "true" ]]; then
        quantize_args+=(
            --finetune_routers
            --rft_epochs "${RFT_EPOCHS:-1}"
            --rft_batch_size "${RFT_BATCH_SIZE:-1}"
            --rft_lr "${RFT_LR:-0.0001}"
            --rft_wd "${RFT_WEIGHT_DECAY:-0.0001}"
        )
    fi
    run_logged "${stage_label}_quantize_avg${target_bit}" "$python_bin" -m gemq.quantize \
        "${common_model_args[@]}" \
        "${common_data_args[@]}" \
        "${quantize_args[@]}"
    "$python_bin" - "$artifact_root" "$stage_label" "$target_bit" "$allocation_path" "$importance_model" "$MODEL_PATH" "${GPTQ_IMPLEMENTATION:-mcmoe}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
stage = sys.argv[2]
bit = sys.argv[3]
allocation = Path(sys.argv[4])
importance_model = sys.argv[5]
quantization_source_model = sys.argv[6]
gptq_implementation = sys.argv[7]
manifest = {
    "stage": stage,
    "target_average_bit": bit,
    "importance_model": importance_model,
    "quantization_source_model": quantization_source_model,
    "allocation": str(allocation),
    "checkpoint": str(root / "checkpoints" / f"avg{bit}"),
    "gptq_implementation": gptq_implementation,
}
path = root / "manifests" / f"{stage}_avg{bit}.json"
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(manifest, indent=2) + "\n")
PY
}

case "$stage" in
    stats)
        stage_label="${3:-base}"
        importance_model="${4:-$MODEL_PATH}"
        IMPORTANCE_MODEL="$importance_model" run_stats "$stage_label" "$importance_model"
        ;;
    allocate)
        if [[ -z "$bit_budget" ]]; then
            echo "allocate requires a bit budget."
            exit 2
        fi
        run_allocate base "$bit_budget"
        ;;
    quantize)
        if [[ -z "$bit_budget" ]]; then
            echo "quantize requires a bit budget."
            exit 2
        fi
        run_quantize base "$bit_budget" "$MODEL_PATH"
        ;;
    bootstrap)
        if [[ -z "$bit_budget" ]]; then
            echo "bootstrap requires a bit budget."
            exit 2
        fi
        IMPORTANCE_MODEL="$MODEL_PATH" run_stats base "$MODEL_PATH"
        run_allocate base "$bit_budget"
        run_quantize base "$bit_budget" "$MODEL_PATH"
        ;;
    progressive)
        previous_bit="${3:-}"
        target_bit="${4:-}"
        if [[ -z "$previous_bit" || -z "$target_bit" ]]; then
            echo "progressive requires <previous-bit> <target-bit>."
            exit 2
        fi
        importance_model="${checkpoint_root}/avg${previous_bit}"
        if [[ ! -d "$importance_model" ]]; then
            echo "Missing previous fake-quantized checkpoint: $importance_model"
            exit 2
        fi
        stage_label="from-${previous_bit}-to-${target_bit}"
        IMPORTANCE_MODEL="$importance_model" run_stats "$stage_label" "$importance_model"
        run_allocate "$stage_label" "$target_bit"
        run_quantize "$stage_label" "$target_bit" "$importance_model"
        ;;
    *)
        echo "Unknown stage: $stage"
        exit 2
        ;;
esac
