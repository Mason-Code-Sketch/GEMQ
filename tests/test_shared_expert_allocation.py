import pickle

import pytest

from gemq.allocate_bits import (
    get_fixed_shared_expert_bits,
    validate_effective_bpe,
    validate_scored_expert_count,
)
from gemq.allocation.ilp_solvers import GEMQSolver
from gemq.utils.model_utils import get_model_info


def _write_scores(tmp_path, num_layers, num_experts, shared_expert_id):
    scores = []
    for _ in range(num_layers):
        layer_scores = {}
        for expert_id in range(num_experts):
            layer_scores[expert_id] = {1: 3.0, 2: 2.0, 3: 1.0}
        layer_scores[shared_expert_id] = {1: -100.0, 2: -50.0, 3: 0.0}
        scores.append(layer_scores)

    path = tmp_path / "scores.pkl"
    with path.open("wb") as handle:
        pickle.dump(scores, handle)
    return path


@pytest.mark.parametrize(
    ("model_name", "num_experts", "target_bpe"),
    [
        ("deepseek-ai/DeepSeek-V2-Lite", 65, 1.5),
        ("deepseek-ai/DeepSeek-V2-Lite", 65, 2.0),
        ("deepseek-ai/DeepSeek-V2-Lite", 65, 2.5),
        ("Qwen/Qwen1.5-MoE-A2.7B", 61, 1.5),
        ("Qwen/Qwen1.5-MoE-A2.7B", 61, 2.0),
        ("Qwen/Qwen1.5-MoE-A2.7B", 61, 2.5),
    ],
)
def test_shared_expert_is_fixed_to_highest_candidate(
    tmp_path, model_name, num_experts, target_bpe
):
    model_info = get_model_info(model_name)
    shared_expert_id = model_info.num_routed_experts_per_layer
    score_path = _write_scores(tmp_path, 2, num_experts, shared_expert_id)
    total_bits = 2 * (
        target_bpe
        * (model_info.num_routed_experts_per_layer + model_info.num_shared_experts_per_layer)
        - (model_info.num_shared_experts_per_layer - 1) * 3
    )

    solver = GEMQSolver(
        score_path,
        x_space=(1, 2, 3),
        extra_constr="c2c3",
        fixed_expert_bits=get_fixed_shared_expert_bits(model_info, (1, 2, 3)),
    )
    validate_scored_expert_count(model_info, solver.num_experts)
    allocation = solver.solve_all(total_bits)

    assert [allocation[layer][shared_expert_id] for layer in allocation] == [3, 3]
    validate_effective_bpe(allocation, model_info, target_bpe, shared_bit=3)


def test_rejects_scores_with_the_wrong_shared_expert_representation():
    model_info = get_model_info("deepseek-ai/DeepSeek-V2-Lite")

    with pytest.raises(ValueError, match="expected 65 scored experts, got 66"):
        validate_scored_expert_count(model_info, 66)
