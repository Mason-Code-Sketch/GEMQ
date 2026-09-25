import os
import os.path as osp
import argparse
import json
import pickle
import re

from gemq.utils.model_utils import get_model_info
from gemq.allocation.ilp_solvers import AVAILABLE_BACKENDS, GEMQSolver
from gemq.resource_ledger import ResourceLedger


def auto_parse_filename(layer_re_path):
    calib_str = ""
    if "math+c4" in layer_re_path:
        calib_str = "MATH+C4"
    elif "c4" in layer_re_path:
        calib_str = "C4"
    elif "math" in layer_re_path:
        calib_str = "MATH"
    else:
        raise ValueError(f"Cannot parse calibration dataset from layer_re_path: {layer_re_path}")

    # extract seed number from layer_re_path
    match = re.search(r'Seed(\d+)', layer_re_path)
    seed_num = match.group(1) if match else "00"
    calib_str += f"-Seed{seed_num}"

    model_str = ""
    if "Uni" in layer_re_path:
        model_str = "_QT"
    elif "QTFT" in layer_re_path:
        model_str = "_QTFT"
    
    return calib_str, model_str


def compute_total_bits(model_name, bpe, bit_cands):
    """
    Auto compute the total bit budget for global ilp.

    Shared experts are fixed at the highest candidate bit. Multiple logical shared
    experts are merged into one physical FFN, so their duplicate logical budget is
    removed before solving over physical expert blocks.
    """
    m = get_model_info(model_name)
    bpl = (
        bpe * (m.num_routed_experts_per_layer + m.num_shared_experts_per_layer) -
        (max(0, m.num_shared_experts_per_layer - 1)) * max(bit_cands)
    )
    return bpl * (m.num_layers - m.first_k_dense_layers)


def get_fixed_shared_expert_bits(model_info, bit_cands):
    """Fix merged shared experts at the bit assumed by the budget formula."""
    if model_info.num_shared_experts_per_layer == 0:
        return {}
    return {model_info.num_routed_experts_per_layer: max(bit_cands)}


def validate_scored_expert_count(model_info, num_scored_experts):
    """Require one scored shared block when the model merges shared experts."""
    expected_scored_experts = (
        model_info.num_routed_experts_per_layer
        + int(model_info.num_shared_experts_per_layer > 0)
    )
    if num_scored_experts != expected_scored_experts:
        raise ValueError(
            "The score file does not match GEMQ's expert representation: "
            f"expected {expected_scored_experts} scored experts, got {num_scored_experts}"
        )


def validate_effective_bpe(opt_set, model_info, target_bpe, shared_bit):
    """Verify that a merged shared block realizes the requested logical bpe."""
    if model_info.num_shared_experts_per_layer == 0:
        return

    shared_expert_id = model_info.num_routed_experts_per_layer
    shared_bits = [experts[shared_expert_id] for experts in opt_set.values()]
    if any(bits != shared_bit for bits in shared_bits):
        raise RuntimeError(
            f"Shared experts must be {shared_bit}-bit, got {shared_bits}"
        )

    physical_bits = sum(sum(experts.values()) for experts in opt_set.values())
    logical_bits = physical_bits + (
        model_info.num_shared_experts_per_layer - 1
    ) * sum(shared_bits)
    logical_experts = len(opt_set) * (
        model_info.num_routed_experts_per_layer
        + model_info.num_shared_experts_per_layer
    )
    effective_bpe = logical_bits / logical_experts
    if abs(effective_bpe - target_bpe) > 1e-9:
        raise RuntimeError(
            f"Allocation realizes {effective_bpe:.12g} bpe, expected {target_bpe:.12g}"
        )
    print(f"Validated effective expert budget: {effective_bpe:.6g} bpe")


def run_gemq_solver(args):
    # parse info
    m = get_model_info(args.model_name)
    bpe = args.bit_budget
    bit_cands = list(map(int, args.bit_candidates.split(",")))

    total_bits = compute_total_bits(args.model_name, bpe, bit_cands)

    # build a solver and solve
    fixed_expert_bits = get_fixed_shared_expert_bits(m, bit_cands)
    global_solver = GEMQSolver(
        layer_re_path=args.layer_re_path,
        x_space=bit_cands,
        extra_constr=args.extra_constr, # NOTE: this args is valid only when using x_space=(1,2,3)
        start_layer_idx=m.first_k_dense_layers,
        backend=args.ilp_backend,
        fixed_expert_bits=fixed_expert_bits,
    )
    validate_scored_expert_count(m, global_solver.num_experts)
    opt_set = global_solver.solve_all(total_bits=total_bits)
    if fixed_expert_bits:
        validate_effective_bpe(
            opt_set, m, bpe, shared_bit=next(iter(fixed_expert_bits.values()))
        )

    # auto generate the save path if not specified
    save_path = args.save_path
    if not save_path:
        bc_str = ",".join(map(str, bit_cands))
        calib_str, model_str = auto_parse_filename(args.layer_re_path)
        const_str = "" if args.extra_constr == "none" else f"_{args.extra_constr}"
        save_path = f"configs/{args.model_name}/GEMQ/{calib_str}_E{bpe:.1f}_B{bc_str}{const_str}{model_str}.pkl"
    
    # save results
    os.makedirs(osp.dirname(save_path), exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump(opt_set, f)
    print("Bit config file saved to:", save_path)


def parse_args():
    parser = argparse.ArgumentParser(description="Bit allocation for MoE models.")
    parser.add_argument(
        "--model_name", type=str, required=True,
        help="Which model to perform bit allocation on",
    )
    parser.add_argument(
        "--layer_re_path", type=str, default="",
        help="Path to the pre-computed weighted layer reconstruction errors",
    )
    parser.add_argument(
        "--bit_budget", type=float, required=True,
        help="Average bits per expert (bpe) budget",
    )
    parser.add_argument(
        "--bit_candidates", type=str, default="1,2,3",
        help="Available bit candidates for allocation, comma-separated string",
    )
    parser.add_argument(
        "--ilp_solver", type=str, required=True, choices=["gemq"],
        help="Bit allocation method (which ILP to formulate), not the numerical solver",
    )
    parser.add_argument(
        "--ilp_backend", type=str, default="highs", choices=list(AVAILABLE_BACKENDS),
        help="Numerical solver used to solve that ILP. 'highs' ships with SciPy and "
             "needs no license; 'gurobi' requires the optional gurobipy extra",
    )
    parser.add_argument(
        "--extra_constr", type=str, default="none",
        help="Whether to enable extra constraints for GEMQ solver",
    )
    parser.add_argument(
        "--save_path", type=str, default="",
        help="Path to save the bit allocation results (leave empty to auto-generate)",
    )
    parser.add_argument(
        "--resource_output", type=str, default="",
        help="Optional JSON path for wall-time and GPU-memory accounting",
    )
    return parser.parse_args()


if __name__ == "__main__":
    # parse args
    args = parse_args()
    print(json.dumps(vars(args), indent=4))
    resource_ledger = ResourceLedger(args.resource_output)

    with resource_ledger.command():
        if args.ilp_solver == "gemq":
            with resource_ledger.component("solve_lp"):
                run_gemq_solver(args)
        else:
            raise ValueError(f"Unknown solver: {args.ilp_solver}")
