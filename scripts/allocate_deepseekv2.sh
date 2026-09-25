#!/bin/bash
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

#  Settings
model_name="deepseek-ai/DeepSeek-V2-Lite"
bits_per_expert=2.0  # target average bits-per-expert
wbits="1,2,3"        # candidate bit-widths
ilp_solver="gemq"    # bit allocation method
ilp_backend="highs"  # ILP solver: "highs" (bundled with scipy) or "gurobi"
extra_constr="c2c3"  # extra constraints for bit allocation
# path to the weighted layer reconstruction errors (i.e., ILP coefficients)
layer_re_path="cache/${model_name}/LayerRE_c4-N128-L2048-Seed0_B1,2,3_faster.pkl"
allocation_path="configs/${model_name}/GEMQ/C4-Seed0_E${bits_per_expert}_B${wbits}_${extra_constr}.pkl"
resource_output="${allocation_path%.pkl}.resource.json"

python -m gemq.allocate_bits \
    --model_name ${model_name} \
    --layer_re_path ${layer_re_path} \
    --bit_budget ${bits_per_expert} \
    --bit_candidates ${wbits} \
    --ilp_solver ${ilp_solver} \
    --ilp_backend ${ilp_backend} \
    --extra_constr ${extra_constr} \
    --save_path ${allocation_path} \
    --resource_output ${resource_output}
