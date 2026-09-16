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

if [[ -z "${MODEL_NAME:-}" || -z "${MODEL_ID:-}" ]]; then
    echo "Protocol config must define MODEL_NAME and MODEL_ID."
    exit 2
fi

resolve_asset_root() {
    local candidate
    for candidate in \
        "${PROTOCOL_ASSET_ROOT:-}" \
        "${repo_root}/../.." \
        "${repo_root}/../../../data"; do
        if [[ -n "$candidate" && -d "${candidate}/models" && -d "${candidate}/datasets" ]]; then
            cd "$candidate" && pwd
            return 0
        fi
    done
    echo "Could not find a parent containing models/ and datasets/." >&2
    exit 2
}

asset_root="$(resolve_asset_root)"
MODEL_PATH="${asset_root}/models/${MODEL_ID}"
DATASET_ROOT="${asset_root}/datasets"

model_key="$(basename "$config_file" .env)"
artifact_root="results/protocol/${model_key}"
stats_root="${artifact_root}/statistics"
alloc_root="${artifact_root}/allocations"
checkpoint_root="${artifact_root}/checkpoints"
evaluation_root="${artifact_root}/evaluations"
python_bin="${PYTHON_BIN:-python}"

# Keep every runtime byproduct of a protocol run under its artifact root.
runtime_root="${artifact_root}/runtime"
mkdir -p "$runtime_root"
export PYTHONDONTWRITEBYTECODE=1
export XDG_CACHE_HOME="${runtime_root}/xdg-cache"
export HF_HOME="${runtime_root}/huggingface"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
export TMPDIR="${runtime_root}/tmp"
mkdir -p "$XDG_CACHE_HOME" "$HF_HOME" "$TMPDIR"

write_invocation_metadata() {
    local metadata_stage="$1"
    shift
    local timestamp
    timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
    mkdir -p "${artifact_root}/metadata" "${artifact_root}/resolved-configs"
    cp "$config_file" "${artifact_root}/resolved-configs/${timestamp}_${metadata_stage}.env"
    "$python_bin" - "${artifact_root}/metadata/${timestamp}_${metadata_stage}.json" \
        "$metadata_stage" "$repo_root" "$config_file" "$@" <<'PY'
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

output = Path(sys.argv[1])
stage = sys.argv[2]
repo = Path(sys.argv[3])
config = Path(sys.argv[4])
arguments = sys.argv[5:]

def command(*args):
    return subprocess.run(args, cwd=repo, text=True, capture_output=True, check=False).stdout.strip()

def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

packages = {}
for package in ("torch", "transformers", "datasets", "scipy", "hqq", "gemlite"):
    try:
        packages[package] = importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        packages[package] = None

tracked = ("gemq/quantize.py", "gemq/compute_model_stats.py", "gemq/allocate_bits.py", "scripts/run_protocol.sh")
output.write_text(json.dumps({
    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    "stage": stage,
    "script_arguments": arguments,
    "protocol_config": str(config),
    "source_commit": command("git", "rev-parse", "HEAD"),
    "source_status": command("git", "status", "--short"),
    "source_sha256": {path: sha256(repo / path) for path in tracked},
    "python": sys.version,
    "platform": platform.platform(),
    "dependencies": packages,
    "runtime_paths": {
        key: os.environ[key]
        for key in ("PYTHONDONTWRITEBYTECODE", "XDG_CACHE_HOME", "HF_HOME", "HF_DATASETS_CACHE", "TMPDIR")
    },
}, indent=2) + "\n")
PY
}

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
    {
        printf '[command] '
        printf '%q ' "$@"
        printf '\n'
    } >> "$log_path"
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
    local stage_checkpoint_root="${checkpoint_root}/avg${target_bit}"
    local gptq_checkpoint="${stage_checkpoint_root}/gptq"
    local router_ft_checkpoint="${stage_checkpoint_root}/router_ft"
    local stage_evaluation_root="${evaluation_root}/${stage_label}/avg${target_bit}"
    mkdir -p "$stage_checkpoint_root" "$stage_evaluation_root"
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
        --save_pre_finetune_path "$gptq_checkpoint"
        --pre_finetune_eval_path "${stage_evaluation_root}/gptq.json"
        --final_eval_path "${stage_evaluation_root}/router_ft.json"
        --save_dtype "${SAVE_DTYPE:-float16}"
        --save_path "$router_ft_checkpoint"
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
    "$python_bin" - "$artifact_root" "$stage_label" "$target_bit" "$allocation_path" "$importance_model" "$MODEL_PATH" "${GPTQ_IMPLEMENTATION:-mcmoe}" "$gptq_checkpoint" "$router_ft_checkpoint" "$stage_evaluation_root" "$MODEL_NAME" "${ILP_BACKEND:-highs}" "${RFT_EPOCHS:-1}" "${RFT_BATCH_SIZE:-1}" "${RFT_LR:-0.0001}" "${RFT_WEIGHT_DECAY:-0.0001}" "${NSAMPLES:-128}" "${SEQLEN:-2048}" <<'PY'
import json
import hashlib
import pickle
import sys
from pathlib import Path

root = Path(sys.argv[1])
stage = sys.argv[2]
bit = sys.argv[3]
allocation = Path(sys.argv[4])
importance_model = sys.argv[5]
quantization_source_model = sys.argv[6]
gptq_implementation = sys.argv[7]
gptq_checkpoint = sys.argv[8]
router_ft_checkpoint = sys.argv[9]
evaluation_root = sys.argv[10]
model_name = sys.argv[11]
ilp_backend = sys.argv[12]
rft_epochs, rft_batch_size, rft_lr, rft_weight_decay = sys.argv[13:17]
nsamples, seqlen = sys.argv[17:19]

with allocation.open("rb") as handle:
    allocation_values = pickle.load(handle)

shared_expert = {"DeepSeek-V2-Lite": (64, 2), "Qwen1.5-MoE-A2.7B": (60, 4)}.get(model_name)
histogram = {}
weighted_bits = 0
weighted_experts = 0
for experts in allocation_values.values():
    for expert_id, bit in experts.items():
        multiplier = shared_expert[1] if shared_expert and expert_id == shared_expert[0] else 1
        histogram[str(bit)] = histogram.get(str(bit), 0) + multiplier
        weighted_bits += bit * multiplier
        weighted_experts += multiplier

allocation_sha256 = hashlib.sha256(allocation.read_bytes()).hexdigest()
manifest = {
    "stage": stage,
    "target_average_bit": bit,
    "importance_model": importance_model,
    "quantization_source_model": quantization_source_model,
    "allocation": str(allocation),
    "allocation_sha256": allocation_sha256,
    "allocation_expert_histogram": histogram,
    "allocation_expert_bpw": weighted_bits / weighted_experts,
    "gptq_checkpoint": gptq_checkpoint,
    "router_ft_checkpoint": router_ft_checkpoint,
    "evaluations": {
        "gptq": str(Path(evaluation_root) / "gptq.json"),
        "router_ft": str(Path(evaluation_root) / "router_ft.json"),
    },
    "gptq_implementation": gptq_implementation,
    "allocation_solver": {"formulation": "gemq", "backend": ilp_backend},
    "data_protocol": {
        "statistics": {"dataset": "c4", "samples": int(nsamples), "sequence_length": int(seqlen)},
        "gptq_and_router_ft": {"dataset": "wikitext2", "samples": int(nsamples), "sequence_length": int(seqlen)},
    },
    "gptq": {
        "candidate_bits": [1, 2, 3], "group_size": 128, "asymmetric": True,
        "mse": True, "reproduce_mcmoe": gptq_implementation == "mcmoe",
        "attention_bits": 4, "dense_bits": 4, "router_bits": 16,
    },
    "router_ft": {
        "enabled": True, "optimizer": "AdamW", "epochs": int(rft_epochs),
        "batch_size": int(rft_batch_size), "learning_rate": float(rft_lr),
        "weight_decay": float(rft_weight_decay),
    },
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
        write_invocation_metadata "stats_${stage_label}" "$stage_label" "$importance_model"
        IMPORTANCE_MODEL="$importance_model" run_stats "$stage_label" "$importance_model"
        ;;
    allocate)
        if [[ -z "$bit_budget" ]]; then
            echo "allocate requires a bit budget."
            exit 2
        fi
        write_invocation_metadata "allocate_base_avg${bit_budget}" "$bit_budget"
        run_allocate base "$bit_budget"
        ;;
    quantize)
        if [[ -z "$bit_budget" ]]; then
            echo "quantize requires a bit budget."
            exit 2
        fi
        write_invocation_metadata "quantize_base_avg${bit_budget}" "$bit_budget" "$MODEL_PATH"
        run_quantize base "$bit_budget" "$MODEL_PATH"
        ;;
    bootstrap)
        if [[ -z "$bit_budget" ]]; then
            echo "bootstrap requires a bit budget."
            exit 2
        fi
        write_invocation_metadata "bootstrap_base_avg${bit_budget}" "$bit_budget" "$MODEL_PATH"
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
        importance_model="${checkpoint_root}/avg${previous_bit}/router_ft"
        if [[ ! -d "$importance_model" ]]; then
            echo "Missing previous fake-quantized checkpoint: $importance_model"
            exit 2
        fi
        stage_label="from-${previous_bit}-to-${target_bit}"
        write_invocation_metadata "progressive_${stage_label}" "$previous_bit" "$target_bit" "$importance_model"
        IMPORTANCE_MODEL="$importance_model" run_stats "$stage_label" "$importance_model"
        run_allocate "$stage_label" "$target_bit"
        run_quantize "$stage_label" "$target_bit" "$importance_model"
        ;;
    *)
        echo "Unknown stage: $stage"
        exit 2
        ;;
esac
