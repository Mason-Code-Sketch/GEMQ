"""Exercise native script outputs without loading the full research models."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest

from gemq.allocate_bits import allocation_output_path


@pytest.mark.parametrize("model", ["deepseek-ai/DeepSeek-V2-Lite", "Qwen/Qwen1.5-MoE-A2.7B", "Qwen/Qwen3-30B-A3B", "mistralai/Mixtral-8x7B-v0.1"])
def test_native_allocation_names_keep_all_six_budgets(model):
    for bit in (1.25, 1.5, 1.75, 2., 2.25, 2.5):
        args = SimpleNamespace(bit_budget=bit, bit_candidates="1,2,3", save_path=None,
                               layer_re_path="LayerRE_c4-N128-L2048-Seed0_B1,2,3_faster.pkl",
                               extra_constr="c2c3", model_name=model)
        assert allocation_output_path(args) == f"configs/{model}/GEMQ/C4-Seed0_E{bit}_B1,2,3_c2c3.pkl"


@pytest.mark.parametrize("model", ["deepseekv2", "qwen1_5_moe", "qwen3moe", "mixtral"])
def test_scripts_keep_distinct_statistics_and_merge_all_phases(tmp_path, model):
    root = Path(__file__).resolve().parents[1]
    repo = tmp_path / "repo"
    (repo / "scripts").mkdir(parents=True)
    for stage in ("compute_stats", "allocate", "quantize"):
        shutil.copyfile(root / f"scripts/{stage}_{model}.sh", repo / f"scripts/{stage}_{model}.sh")
    # Native modules are replaced only at the process boundary. Actual ledgers,
    # filenames and shell scripts run, including their inline merge commands.
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    executable = bin_dir / "python"
    executable.write_text(f"#!{sys.executable}\n" + r'''
import os, sys
from pathlib import Path
from types import SimpleNamespace
from gemq.allocate_bits import allocation_output_path
from gemq.resource_ledger import ResourceLedger
if sys.argv[1] == "-":
    sys.argv = sys.argv[1:]
    exec(compile(sys.stdin.read(), "<script-merge>", "exec"))
    raise SystemExit(0)
args = sys.argv[3:]
def value(key, default=None):
    return args[args.index(key)+1] if key in args else default
module = sys.argv[2]
if module == "gemq.compute_model_stats":
    mode = value("--mode")
    target = value("--layer_grads_path" if mode == "layer_grads" else "--layer_re_path")
    path = value("--resource_output", str(Path(target).with_suffix(".resource.json")))
    names = ["comp_grad" if mode == "layer_grads" else "comp_stats"]
elif module == "gemq.allocate_bits":
    allocation = allocation_output_path(SimpleNamespace(bit_budget=float(value("--bit_budget")), bit_candidates=value("--bit_candidates"), save_path=value("--save_path"), layer_re_path=value("--layer_re_path"), extra_constr=value("--extra_constr"), model_name=value("--model_name")))
    path = value("--resource_output", str(Path(allocation).with_suffix(".resource.json")))
    names = ["solve_lp"]
else:
    path = value("--resource_output")
    assert path, "quantization script must explicitly locate its resource output"
    names = ["quant_gptq"] + (["ft_routers"] if "--finetune_routers" in args else [])
ledger = ResourceLedger(path)
with ledger.command():
    for name in names:
        with ledger.component(name):
            pass
''')
    executable.chmod(0o755)
    env = dict(os.environ, PATH=f"{bin_dir}:{os.environ['PATH']}", PYTHON=str(executable),
               PYTHONPATH=str(root), PYTHONDONTWRITEBYTECODE="1")
    for stage in ("compute_stats", "allocate", "quantize"):
        result = subprocess.run(["bash", str(repo / f"scripts/{stage}_{model}.sh")],
                                cwd=tmp_path, env=env, capture_output=True, text=True)
        assert result.returncode == 0, result.stdout + result.stderr
    grads = list((repo / "cache").rglob("LayerGrads*.resource.json"))
    stats = list((repo / "cache").rglob("LayerRE*.resource.json"))
    assert len(grads) == len(stats) == 1
    assert set(json.loads(grads[0].read_text())["components"]) == {"comp_grad"}
    assert set(json.loads(stats[0].read_text())["components"]) == {"comp_grad", "comp_stats"}
    reports = list((repo / "results").rglob("resource_breakdown.json"))
    assert len(reports) == 1
    result = json.loads(reports[0].read_text())
    assert result["missing_sources"] == []
    assert {"comp_grad", "comp_stats", "solve_lp", "quant_gptq"} <= set(result["components"])
    assert not (tmp_path / "cache").exists()
