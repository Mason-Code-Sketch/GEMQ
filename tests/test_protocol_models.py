"""Static coverage for local model identifiers and protocol configs."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from gemq.utils.model_registry import NAME_TO_MODEL, ModelType


class ProtocolModelNamesTest(unittest.TestCase):
    def test_local_protocol_models_are_registered(self):
        self.assertEqual(NAME_TO_MODEL["DeepSeek-V2-Lite"], ModelType.DEEPSEEKV2)
        self.assertEqual(NAME_TO_MODEL["Qwen1.5-MoE-A2.7B"], ModelType.QWEN2MOE)
        self.assertEqual(NAME_TO_MODEL["Qwen3-30B-A3B-Base"], ModelType.QWEN3MOE)
        self.assertEqual(NAME_TO_MODEL["Mixtral-8x7B-v0.1"], ModelType.MIXTRAL)

    def test_protocol_configs_use_relative_model_and_dataset_paths(self):
        config_dir = Path(__file__).parents[1] / "configs" / "protocol"
        for config_path in config_dir.glob("*.env"):
            values = dict(
                line.split("=", 1)
                for line in config_path.read_text().splitlines()
                if line and not line.startswith("#")
            )
            self.assertTrue(values["MODEL_PATH"].startswith("../../../data/models/"))
            self.assertEqual(values["DATASET_ROOT"], "../../../data/datasets")
            self.assertEqual(values["EXPERIMENT_PROTOCOL"], "vivit_ggn")
            self.assertEqual(values["GPTQ_IMPLEMENTATION"], "mcmoe")

    def test_protocol_paths_exist_from_runner_root(self):
        repo_root = Path(__file__).parents[1]
        config_dir = repo_root / "configs" / "protocol"
        for config_path in config_dir.glob("*.env"):
            values = dict(
                line.split("=", 1)
                for line in config_path.read_text().splitlines()
                if line and not line.startswith("#")
            )
            self.assertTrue((repo_root / values["MODEL_PATH"]).is_dir(), config_path)
            self.assertTrue((repo_root / values["DATASET_ROOT"]).is_dir(), config_path)

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
                "MODEL_PATH=base-model\n"
                "DATASET_ROOT=datasets\n"
                "PYTHON_BIN=" + str(fake_python) + "\n"
                "GPTQ_IMPLEMENTATION=mcmoe\n"
            )
            try:
                bootstrap = subprocess.run(
                    [str(runner), str(env_file), "bootstrap", "3"],
                    cwd=repo_root,
                    env={**os.environ},
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(bootstrap.returncode, 0, bootstrap.stderr)
                progressive = subprocess.run(
                    [str(runner), str(env_file), "progressive", "3", "2.5"],
                    cwd=repo_root,
                    env={**os.environ},
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
                self.assertEqual(manifest["quantization_source_model"], "base-model")
                self.assertEqual(manifest["gptq_implementation"], "mcmoe")
            finally:
                shutil.rmtree(artifact_root, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
