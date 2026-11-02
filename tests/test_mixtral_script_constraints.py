"""Run the actual Mixtral shell scripts and compare allocation/consumer paths."""

import ast
import json
import os
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


def native_allocation_output_path(root, args):
    # These pure helpers have no model dependency. Load their actual source so
    # the shell contract can also run on a machine without the GPU ML runtime.
    path = root / "gemq/allocate_bits.py"
    module = ast.parse(path.read_text())
    nodes = [node for node in module.body if isinstance(node, ast.FunctionDef)
             and node.name in {"auto_parse_filename", "allocation_output_path"}]
    assert len(nodes) == 2
    scope = {"re": re}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), scope)
    return scope["allocation_output_path"](args)


class MixtralScriptConstraintTest(unittest.TestCase):
    def test_all_six_budgets_select_matching_constraints_and_paths(self):
        root = Path(__file__).resolve().parents[1]
        for bit in (1.25, 1.5, 1.75, 2.0, 2.25, 2.5):
            with self.subTest(bit=bit), tempfile.TemporaryDirectory() as folder:
                tmp = Path(folder)
                repo = tmp / "repo"
                scripts = repo / "scripts"
                scripts.mkdir(parents=True)
                capture = tmp / "calls.jsonl"
                bin_dir = tmp / "bin"
                bin_dir.mkdir()
                executable = bin_dir / "python"
                executable.write_text(f"#!{sys.executable}\n" +
                    "import json,os,sys\n"
                    "with open(os.environ['CAPTURE_FILE'],'a') as f:\n"
                    " f.write(json.dumps({'args':sys.argv[1:],'cwd':os.getcwd()})+'\\n')\n")
                executable.chmod(0o755)
                env = dict(os.environ, PATH=f"{bin_dir}:{os.environ['PATH']}",
                           CAPTURE_FILE=str(capture), PYTHONDONTWRITEBYTECODE="1")
                for stage, setting in (("allocate", "bits_per_expert"), ("quantize", "bpe")):
                    original = (root / f"scripts/{stage}_mixtral.sh").read_text()
                    text, count = re.subn(rf"^{setting}=2\.0(?=\s)",
                                          f"{setting}={bit}", original, count=1,
                                          flags=re.MULTILINE)
                    self.assertEqual(count, 1)
                    path = scripts / f"{stage}_mixtral.sh"
                    path.write_text(text)
                    result = subprocess.run(["bash", str(path)], cwd=tmp, env=env,
                                            capture_output=True, text=True)
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                calls = [json.loads(line) for line in capture.read_text().splitlines()]
                self.assertTrue(all(Path(row["cwd"]).resolve() == repo.resolve() for row in calls))
                allocate = next(row["args"][2:] for row in calls
                                if row["args"][:2] == ["-m", "gemq.allocate_bits"])
                quantize = next(row["args"][2:] for row in calls
                                if row["args"][:2] == ["-m", "gemq.quantize"])

                def value(args, name):
                    return args[args.index(name) + 1]

                constraint = "none" if bit == 1.25 else "c2c3"
                self.assertEqual(value(allocate, "--extra_constr"), constraint)
                self.assertEqual(float(value(allocate, "--bit_budget")), bit)
                generated = native_allocation_output_path(root, SimpleNamespace(
                    model_name=value(allocate, "--model_name"), bit_budget=bit,
                    bit_candidates=value(allocate, "--bit_candidates"), save_path=None,
                    layer_re_path=value(allocate, "--layer_re_path"), extra_constr=constraint))
                self.assertEqual(value(quantize, "--bit_cfg"), generated)
                self.assertIn("--reproduce_mcmoe", quantize)
                self.assertIn("--mse", quantize)
                self.assertEqual(value(quantize, "--groupsize"), "128")


if __name__ == "__main__":
    unittest.main()
