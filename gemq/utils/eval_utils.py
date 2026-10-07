import gc
import math
from importlib.metadata import version
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset

from gemq.utils.model_utils import get_blocks, move_embed, move_head


def get_testenc(tokenizer, dataset, seqlen):
    if dataset == "wikitext2":
        testdata = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        testenc = tokenizer("\n\n".join(testdata["text"]), return_tensors="pt")

    elif dataset == "c4":
        testdata = load_dataset("allenai/c4", data_files={"validation": "en/c4-validation.00000-of-00008.json.gz"}, split="validation")
        testenc = tokenizer(" ".join(testdata[:1100]["text"]), return_tensors="pt")
        testenc = testenc.input_ids[:, :(256 * seqlen)]

        class TokenizerWrapper:
            def __init__(self, input_ids):
                self.input_ids = input_ids
        testenc = TokenizerWrapper(testenc)

    else:
        raise NotImplementedError(f"Dataset {dataset} not implemented.")

    return testenc


def compute_perplexity(model, input_ids, dataset_name) -> float:
    """
    Compute the perplexity of the model on the given dataset.
    """
    nlls = []
    nsamples = input_ids.numel() // model.seqlen
    for i in tqdm(range(nsamples), desc=f"Evaluating [{dataset_name}]"):
        batch = input_ids[:, (i * model.seqlen) : ((i + 1) * model.seqlen)].to(model.device)
        lm_logits = model(batch).logits
        shift_logits = lm_logits[:, :-1, :].contiguous().float()
        shift_labels = input_ids[:, (i * model.seqlen) : ((i + 1) * model.seqlen)][:, 1:].to(model.device)
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        neg_log_likelihood = loss.float() * model.seqlen
        nlls.append(neg_log_likelihood)

    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen)).item()
    return ppl


def compute_perplexity_offload(model, model_name, input_ids, dataset_name):
    """
    Compute the perplexity of the model on the given dataset.
    This function uses dynamic weights offloading for memory-efficient evaluation.
    """
    # disable kv cache since we are running batch generation for evaluation
    use_cache = model.config.use_cache
    model.config.use_cache = False

    nsamples = input_ids.numel() // model.seqlen

    # retrieve blocks that require quantization
    layers = get_blocks(model, model_name)

    # get input and kwargs to the first layer decoding layer
    inps = []
    layer_kwargs = {}

    move_embed(model, model_name, "cuda")
    layers[0] = layers[0].to("cuda")
    
    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

            if hasattr(self.module, "attention_type"):
                self.attention_type = self.module.attention_type

        def forward(self, inp, **kwargs):
            inps.append(inp)  # NOTE: inp is (bsz, seqlen, hidden_size)
            layer_kwargs.update(kwargs)
            raise ValueError  # early exit to break later inference

    layers[0] = Catcher(layers[0])
    for i in range(nsamples):
        batch = input_ids[:, (i * model.seqlen) : ((i + 1) * model.seqlen)].to("cuda")
        try:
            model(batch)
        except ValueError:
            pass
    layers[0] = layers[0].module  # restore
    inps = torch.cat(inps, dim=0)  # (nsamples, seqlen, hidden_size)
    
    # for memory savings
    move_embed(model, model_name, "cpu")
    layers[0] = layers[0].cpu()
    gc.collect()
    torch.cuda.empty_cache()


    # forward pass with dynamic offloading
    outs = torch.zeros_like(inps)
    for i in tqdm(range(len(layers)), desc=f"Evaluating [{dataset_name}]"):
        layer = layers[i].to("cuda")
        for j in range(nsamples):
            outs[j] = layer(inps[j: j+1], **layer_kwargs)[0]
        layers[i] = layer.cpu()
        gc.collect()
        torch.cuda.empty_cache()
        inps, outs = outs, inps
    move_head(model, model_name, "cuda")

    # compute perplexity
    nlls = []
    nsamples = input_ids.numel() // model.seqlen
    for i in range(nsamples):
        hidden_states = model.model.norm(inps[i:i + 1])
        lm_logits = model.lm_head(hidden_states)
        shift_logits = lm_logits[:, :-1, :].contiguous().float()
        shift_labels = input_ids[:, (i * model.seqlen) : ((i + 1) * model.seqlen)][:, 1:].to("cuda")
        # loss_fct = nn.CrossEntropyLoss()
        loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        neg_log_likelihood = loss.float() * model.seqlen
        nlls.append(neg_log_likelihood)
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen)).item()

    model.config.use_cache = use_cache  # restore

    return ppl


@torch.inference_mode()
def evaluate_perplexity(model, tokenizer, datasets, model_name, offload=True):
    """
    Evaluate the model on a given dataset.
    """
    # NOTE: disable kv cache since we are running batch generation for evaluation
    use_cache = model.config.use_cache
    model.config.use_cache = False

    # for each dataset
    for dataset in datasets:
        testenc = get_testenc(tokenizer, dataset, model.seqlen)
        if offload:
            ppl = compute_perplexity_offload(model, model_name, testenc.input_ids, dataset)
        else:
            ppl = compute_perplexity(model, testenc.input_ids, dataset)
        print(f"[{dataset}] ppl: {ppl:.4f}")

    # restore
    model.config.use_cache = use_cache



ZEROSHOT_TASKS = (
    "piqa", "arc_easy", "arc_challenge", "hellaswag", "winogrande", "mathqa", "mmlu",
)
LM_EVAL_VERSION = "0.4.13"


def summarize_lm_eval(results, tasks, include_gsm8k=False):
    scores = {}
    for task in tasks:
        metrics = results.get("results", {}).get(task, {})
        if task == "mmlu" and "acc,none" not in metrics:
            metrics = results.get("groups", {}).get(task, {})
        metric = next((key for key in ("acc_norm,none", "acc,none") if key in metrics), None)
        if metric is None:
            raise ValueError(f"Missing accuracy for task {task}")
        value = float(metrics[metric])
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"Invalid accuracy for task {task}: {value}")
        scores[task] = {"metric": metric, "value": value}
    summary = {"tasks": scores}
    if set(tasks) == set(ZEROSHOT_TASKS):
        summary["seven_task_average"] = sum(item["value"] for item in scores.values()) / len(ZEROSHOT_TASKS)
    if include_gsm8k:
        metrics = results.get("results", {}).get("gsm8k", {})
        exact_match = {key: float(value) for key, value in metrics.items() if key.startswith("exact_match,")}
        if not exact_match or any(not math.isfinite(value) or not 0 <= value <= 1 for value in exact_match.values()):
            raise ValueError("Missing or invalid GSM8K exact-match metrics")
        summary["gsm8k"] = exact_match
    return summary


def run_lm_eval(
    model, tokenizer, tasks=None, batch_size=1, num_fewshot=0,
    limit=None, seed=0, max_length=None, include_gsm8k=False,
):
    from lm_eval import evaluator
    from lm_eval.models.huggingface import HFLM

    if version("lm_eval") != LM_EVAL_VERSION:
        raise RuntimeError(f"Install lm-eval[hf]=={LM_EVAL_VERSION} for this evaluation")
    tasks = list(tasks if tasks is not None else ("mmlu",))
    if not tasks or len(set(tasks)) != len(tasks):
        raise ValueError("Tasks must be nonempty and unique")
    requested = tasks + (["gsm8k"] if include_gsm8k else [])
    wrapped = HFLM(
        pretrained=model, tokenizer=tokenizer, backend="causal",
        batch_size=batch_size, max_length=max_length,
    )
    results = evaluator.simple_evaluate(
        model=wrapped, tasks=requested, num_fewshot=num_fewshot,
        limit=limit, log_samples=False, apply_chat_template=False,
        random_seed=seed, numpy_random_seed=seed, torch_random_seed=seed,
        fewshot_random_seed=seed,
    )
    if results is None:
        raise RuntimeError("Downstream evaluation returned no results")
    summary = summarize_lm_eval(results, tasks, include_gsm8k)
    for task, score in summary["tasks"].items():
        print(f"{task:<25}: {score['value'] * 100:.2f} (%) [{score['metric']}]")
    if "seven_task_average" in summary:
        print(f"Avg (7 tasks): {summary['seven_task_average'] * 100:.2f} (%)")
    for metric, value in summary.get("gsm8k", {}).items():
        print(f"gsm8k {metric}: {value * 100:.2f} (%)")
    return {"summary": summary, "harness": results}
