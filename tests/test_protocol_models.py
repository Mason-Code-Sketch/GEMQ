"""Static coverage for local model identifiers and protocol configs."""

import json
import os
from pathlib import Path
import pickle
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import torch

from gemq.allocation.ilp_solvers import GEMQSolver
from gemq.quantizers.gptq import MCMoeGPTQWeightQuantizer
from gemq.utils.model_registry import NAME_TO_MODEL, ModelType


class ProtocolModelNamesTest(unittest.TestCase):
    def test_local_protocol_models_are_registered(self):
        self.assertEqual(NAME_TO_MODEL["DeepSeek-V2-Lite"], ModelType.DEEPSEEKV2)
        self.assertEqual(NAME_TO_MODEL["Qwen1.5-MoE-A2.7B"], ModelType.QWEN2MOE)
        self.assertEqual(NAME_TO_MODEL["Qwen3-30B-A3B-Base"], ModelType.QWEN3MOE)
        self.assertEqual(NAME_TO_MODEL["Mixtral-8x7B-v0.1"], ModelType.MIXTRAL)

    def test_protocol_configs_identify_models_without_host_paths(self):
        config_dir = Path(__file__).parents[1] / "configs" / "protocol"
        for config_path in config_dir.glob("*.env"):
            values = dict(
                line.split("=", 1)
                for line in config_path.read_text().splitlines()
                if line and not line.startswith("#")
            )
            self.assertIn("MODEL_ID", values)
            self.assertNotIn("MODEL_PATH", values)
            self.assertNotIn("DATASET_ROOT", values)
            self.assertEqual(values["EXPERIMENT_PROTOCOL"], "vivit_ggn")
            self.assertEqual(values["GPTQ_IMPLEMENTATION"], "mcmoe")

    def test_deepseek_protocol_uses_native_bfloat16(self):
        config_path = Path(__file__).parents[1] / "configs" / "protocol" / "deepseek-v2-lite.env"
        values = dict(
            line.split("=", 1)
            for line in config_path.read_text().splitlines()
            if line and not line.startswith("#")
        )
        self.assertEqual(values["MODEL_DTYPE"], "bfloat16")
        self.assertEqual(values["SAVE_DTYPE"], "bfloat16")

    def test_runner_resolves_asset_root_at_runtime(self):
        runner = (Path(__file__).parents[1] / "scripts" / "run_protocol.sh").read_text()
        self.assertIn("resolve_asset_root", runner)
        self.assertIn("PROTOCOL_ASSET_ROOT", runner)
        self.assertIn('MODEL_PATH="${asset_root}/models/${MODEL_ID}"', runner)

    def test_runner_preserves_progressive_model_roles(self):
        runner = (Path(__file__).parents[1] / "scripts" / "run_protocol.sh").read_text()
        self.assertIn("--reproduce_mcmoe", runner)
        self.assertIn("--skip_pre_finetune_eval", runner)
        self.assertIn('importance_model="${checkpoint_root}/avg${previous_bit}"', runner)
        self.assertIn('run_quantize "$stage_label" "$target_bit" "$importance_model"', runner)

    def test_bootstrap_and_progressive_manifests(self):
        repo_root = Path(__file__).parents[1]
        runner = repo_root / "scripts" / "run_protocol.sh"
        model_key = "protocol-runner-test"
        artifact_root = repo_root / "results" / "protocol" / model_key
        shutil.rmtree(artifact_root, ignore_errors=True)
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            fake_python = temporary / "fake_python"
            fake_python.write_text(
                "#!/bin/sh\n"
                "if [ \"$1\" = '-' ]; then exec python \"$@\"; fi\n"
                "output=''\n"
                "previous=''\n"
                "for arg in \"$@\"; do\n"
                "  if [ \"$previous\" = '--layer_grads_path' ] || [ \"$previous\" = '--layer_re_path' ] || [ \"$previous\" = '--save_path' ]; then output=\"$arg\"; fi\n"
                "  previous=\"$arg\"\n"
                "done\n"
                "case \"$*\" in\n"
                "  *gemq.quantize*) mkdir -p \"$output\" ;;\n"
                "  *) mkdir -p \"$(dirname \"$output\")\"; : > \"$output\" ;;\n"
                "esac\n"
            )
            fake_python.chmod(0o755)
            env_file = temporary / f"{model_key}.env"
            env_file.write_text(
                "MODEL_NAME=DeepSeek-V2-Lite\n"
                "MODEL_ID=base-model\n"
                "PYTHON_BIN=" + str(fake_python) + "\n"
                "GPTQ_IMPLEMENTATION=mcmoe\n"
            )
            assets = temporary / "assets"
            (assets / "models" / "base-model").mkdir(parents=True)
            (assets / "datasets").mkdir()
            try:
                bootstrap = subprocess.run(
                    [str(runner), str(env_file), "bootstrap", "3"],
                    cwd=repo_root,
                    env={**os.environ, "PROTOCOL_ASSET_ROOT": str(assets)},
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(bootstrap.returncode, 0, bootstrap.stderr)
                progressive = subprocess.run(
                    [str(runner), str(env_file), "progressive", "3", "2.5"],
                    cwd=repo_root,
                    env={**os.environ, "PROTOCOL_ASSET_ROOT": str(assets)},
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(progressive.returncode, 0, progressive.stderr)
                manifest = json.loads(
                    (artifact_root / "manifests" / "from-3-to-2.5_avg2.5.json").read_text()
                )
                self.assertEqual(
                    manifest["importance_model"],
                    f"results/protocol/{model_key}/checkpoints/avg3",
                )
                self.assertEqual(
                    manifest["quantization_source_model"],
                    str(assets / "models" / "base-model"),
                )
                self.assertEqual(manifest["gptq_implementation"], "mcmoe")
            finally:
                shutil.rmtree(artifact_root, ignore_errors=True)

    def test_shared_expert_is_fixed_to_highest_candidate_bit(self):
        coefficients = {
            0: {
                0: {1: 3.0, 2: 2.0, 3: 1.0},
                1: {1: 3.0, 2: 2.0, 3: 1.0},
                2: {1: 0.0, 2: 0.0, 3: 100.0},
            }
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "layer_re.pkl"
            with path.open("wb") as handle:
                pickle.dump(coefficients, handle)
            solver = GEMQSolver(
                layer_re_path=path,
                x_space=(1, 2, 3),
                backend="highs",
                fixed_expert_bits={2: 3},
            )
            allocation = solver.solve_all(total_bits=6)
        self.assertEqual(allocation[0][2], 3)

    def test_mcmoe_mse_search_uses_fp32_for_bf16_weights(self):
        for dtype in (torch.bfloat16, torch.float16):
            quantizer = MCMoeGPTQWeightQuantizer(
                torch.randn(4, 8, dtype=dtype), nbits=2, groupsize=4, mse=True
            )
            seen_scales = []
            original = quantizer.quantize_vector

            def traced(values, scales, zeros, max_int):
                seen_scales.append(scales.detach().clone())
                return original(values, scales, zeros, max_int)

            with patch.object(quantizer, "quantize_vector", side_effect=traced):
                scales, zeros, _max_int = quantizer.find_params(quantizer.W)
            self.assertEqual(scales.dtype, torch.float32)
            self.assertEqual(zeros.dtype, torch.float32)
            self.assertEqual(len(seen_scales), 101)
            self.assertEqual(torch.unique(torch.stack(seen_scales), dim=0).shape[0], 101)


if __name__ == "__main__":
    unittest.main()
