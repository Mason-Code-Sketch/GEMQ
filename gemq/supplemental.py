"""Fixed, independent MMLU 5-shot and GSM8K 0-shot evaluation protocols."""
from __future__ import annotations

import hashlib
import json
import math
from importlib.metadata import version

PROFILES = {"mmlu_5shot": ("mmlu", 5), "gsm8k_0shot": ("gsm8k", 0)}
HARNESS_VERSION = "0.4.13"
# HFLM's default generation budget, made explicit for reproducibility.
GSM8K_MAX_NEW_TOKENS = 256


def summarize(raw, profile, *, limit=None):
    """Reject wrong shots, incomplete full evaluations and invalid metrics."""
    task, shots = PROFILES[profile]
    shot_map = raw.get("n-shot", {})
    leaves = [name for name in raw.get("configs", {})
              if name.startswith("mmlu_")] if task == "mmlu" else ["gsm8k"]
    if not leaves or any(shot_map.get(name) != shots for name in leaves):
        raise ValueError(f"{profile}: missing or inconsistent actual n-shot metadata")
    counts = {}
    for name in leaves:
        count = raw.get("n-samples", {}).get(name, {}).get("effective")
        if not isinstance(count, int) or count <= 0:
            raise ValueError(f"{profile}: missing effective sample count for {name}")
        counts[name] = count
    if limit is None:
        expected = 14042 if task == "mmlu" else 1319
        if sum(counts.values()) != expected or (task == "mmlu" and len(leaves) != 57):
            raise ValueError(f"{profile}: incomplete full test set")
    if task == "gsm8k":
        metrics = raw["results"][task]
        keys = ["exact_match,strict-match", "exact_match,flexible-extract"]
        if not all(key in metrics for key in keys):
            raise ValueError("GSM8K requires both strict-match and flexible-extract")
        scores = {key: float(metrics[key]) for key in keys}
        if any(not math.isfinite(v) or not 0 <= v <= 1 for v in scores.values()):
            raise ValueError("invalid GSM8K exact-match score")
        return {"gsm8k": scores, "sample_counts": counts}
    subjects = {name: float(raw["results"][name]["acc,none"]) for name in leaves}
    if any(not math.isfinite(v) or not 0 <= v <= 1 for v in subjects.values()):
        raise ValueError("invalid MMLU accuracy")
    mean = sum(subjects[name] * counts[name] for name in leaves) / sum(counts.values())
    group = raw.get("groups", {}).get("mmlu", raw.get("results", {}).get("mmlu", {}))
    if "acc,none" not in group or not math.isclose(float(group["acc,none"]), mean, abs_tol=1e-10):
        raise ValueError("MMLU aggregate does not match sample-weighted subject accuracy")
    return {"mmlu": {"metric": "acc,none", "value": mean},
            "subjects": subjects, "sample_counts": counts}


def evaluate(model, tokenizer, profile, *, seed=0, batch_size="auto",
             max_batch_size=None, max_length=None, limit=None):
    """Use the branch's already-loaded model, without changing its weight backend."""
    if version("lm_eval") != HARNESS_VERSION:
        raise RuntimeError(f"supplemental evaluation requires lm_eval=={HARNESS_VERSION}")
    from lm_eval import evaluator
    from lm_eval.models.huggingface import HFLM

    task, shots = PROFILES[profile]
    if max_batch_size is None:
        max_batch_size = {"mmlu_5shot": 19, "gsm8k_0shot": 107}[profile]
    model.eval()
    wrapped = HFLM(pretrained=model, tokenizer=tokenizer, backend="causal",
                   batch_size=batch_size, max_batch_size=max_batch_size,
                   max_length=max_length, add_bos_token=False, parallelize=False)
    batch_probes = []
    if task == "gsm8k":
        from gemq.generation_batch import install_generation_batch_probe
        batch_probes = install_generation_batch_probe(wrapped, max_new_tokens=GSM8K_MAX_NEW_TOKENS)
    generation = {"do_sample": False, "temperature": 0.0,
                  "max_gen_toks": GSM8K_MAX_NEW_TOKENS} if task == "gsm8k" else None
    kwargs = {"gen_kwargs": generation} if generation is not None else {}
    config = getattr(model, "config", None)
    disable_cache = getattr(config, "model_type", None) == "deepseek_v2"
    previous_cache = getattr(config, "use_cache", None)
    if disable_cache:
        config.use_cache = False
    try:
        raw = evaluator.simple_evaluate(
            model=wrapped, tasks=[task], num_fewshot=shots, limit=limit,
            apply_chat_template=False, log_samples=False,
            random_seed=seed, numpy_random_seed=seed, torch_random_seed=seed,
            fewshot_random_seed=seed, **kwargs,
        )
    finally:
        if disable_cache:
            config.use_cache = previous_cache
    if raw is None:
        raise RuntimeError("supplemental evaluator returned no results")
    summary = summarize(raw, profile, limit=limit)
    configs = raw.get("configs", {})
    if not configs:
        raise ValueError("harness did not return task configurations")
    canonical = json.dumps(configs, sort_keys=True, default=str).encode()
    return {"summary": summary, "harness": raw, "protocol": {
        "profile": profile, "task": task, "num_fewshot": shots,
        "seed": seed, "limit": limit, "status": "smoke" if limit is not None else "complete",
        "add_bos_token": False, "apply_chat_template": False,
        "batch_size": batch_size, "max_batch_size": max_batch_size,
        "max_length": max_length, "generation_kwargs": generation,
        "generation_batch_probes": batch_probes,
        "task_configs_sha256": hashlib.sha256(canonical).hexdigest(),
        "versions": {name: version(name) for name in ("lm_eval", "transformers", "torch", "datasets")},
    }}
