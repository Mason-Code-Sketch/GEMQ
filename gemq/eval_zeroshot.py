"""Evaluate an existing GEMQ checkpoint without quantization or Router training."""

import argparse
import json
import sys
import traceback
from contextlib import ExitStack, redirect_stderr, redirect_stdout
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from gemq.utils.eval_utils import LM_EVAL_VERSION, ZEROSHOT_TASKS, run_lm_eval
from gemq.utils.hf_loading import (
    align_deepseek_softmax_scale, describe_model_impl, load_quantized_model,
)


class _Tee:
    def __init__(self, terminal, log):
        self.terminal = terminal
        self.log = log

    def write(self, text):
        self.terminal.write(text)
        self.log.write(text)
        return len(text)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def isatty(self):
        return False


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def batch_size(value):
    return value if value == "auto" else positive_int(value)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_path", type=Path, required=True)
    parser.add_argument("--is_fp", action="store_true", help="Load a floating-point HF checkpoint")
    parser.add_argument("--trust_remote_code", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--model_dtype", choices=("float16", "bfloat16", "float32"), default="float16")
    parser.add_argument("--backend", choices=("auto", "pytorch", "triton"), default="auto",
                        help="auto uses validated GEMQ kernels for packed Qwen3 FP16 on CUDA")
    parser.add_argument("--tasks", nargs="+", choices=ZEROSHOT_TASKS, default=list(ZEROSHOT_TASKS))
    parser.add_argument("--batch_size", type=batch_size, default=1)
    parser.add_argument("--max_batch_size", type=positive_int, default=8)
    parser.add_argument("--max_length", type=positive_int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--include_gsm8k", action="store_true")
    parser.add_argument("--limit", type=positive_int, help="Terminal-only check: examples per leaf task")
    parser.add_argument("--output", type=Path, help="New JSON path; defaults to MODEL_PATH/zeroshot.json")
    args = parser.parse_args(argv)
    if len(set(args.tasks)) != len(args.tasks):
        parser.error("--tasks must not contain duplicates")
    if args.limit is not None and args.output is not None:
        parser.error("--limit reports to the terminal and cannot be combined with --output")
    return args


def result_path(args):
    args.model_path = args.model_path.expanduser().resolve()
    if not args.model_path.is_dir():
        raise FileNotFoundError(args.model_path)
    required = ["config.json"] + ([] if args.is_fp else ["qmodel.pt"])
    for name in required:
        if not (args.model_path / name).is_file():
            raise FileNotFoundError(args.model_path / name)
    if args.limit is not None:
        return None
    output = (args.output or args.model_path / "zeroshot.json").expanduser().resolve()
    if output.suffix != ".json":
        raise ValueError("--output must be a JSON path")
    for path in (output, output.with_suffix(".log")):
        if path.exists():
            raise FileExistsError(f"Evaluation output already exists: {path}")
    return output


def load_model(args):
    tokenizer = AutoTokenizer.from_pretrained(
        str(args.model_path), trust_remote_code=args.trust_remote_code,
    )
    dtype = getattr(torch, args.model_dtype)
    if args.is_fp:
        model = AutoModelForCausalLM.from_pretrained(
            str(args.model_path), torch_dtype=dtype, device_map=args.device,
            trust_remote_code=args.trust_remote_code,
        )
    else:
        model = load_quantized_model(
            str(args.model_path), compute_dtype=dtype, device=args.device,
            trust_remote_code=args.trust_remote_code,
        )
    align_deepseek_softmax_scale(model)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    args.backend = prepare_backend(model, tokenizer, args)
    return model, tokenizer


@torch.inference_mode()
def prepare_backend(model, tokenizer, args):
    supported = (
        not args.is_fp and model.config.model_type == "qwen3_moe"
        and args.device.startswith("cuda") and args.model_dtype == "float16"
    )
    if supported:
        experts = [layer.mlp.experts for layer in model.model.layers if hasattr(layer.mlp, "experts")]
        supported = bool(experts) and all(isinstance(group, torch.nn.ModuleList) for group in experts)
    backend = ("triton" if supported else "pytorch") if args.backend == "auto" else args.backend
    if backend == "pytorch":
        return backend
    if not supported:
        raise ValueError("The Triton evaluation backend requires a packed Qwen3 FP16 model "
                         "on CUDA with ModuleList experts")

    from gemq.inference.patch import prepare_for_inference

    # Check both single-sequence and batched prefill before evaluating any tasks.
    device = model.get_input_embeddings().weight.device
    tokens = tokenizer("The following text is used to check quantized inference.",
                       return_tensors="pt").input_ids.to(device)
    inputs = [tokens.repeat(batch, (length + tokens.shape[1] - 1) // tokens.shape[1])[:, :length]
              for batch, length in ((1, 32), (2, 128))]
    references = []
    for ids in inputs:
        torch.cuda.synchronize()
        start = perf_counter()
        logits = model(ids, use_cache=False).logits
        torch.cuda.synchronize()
        references.append((logits.cpu(), perf_counter() - start))
        del logits

    prepare_for_inference(model, "Qwen/Qwen3-30B-A3B", is_fp=False)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for ids, (reference, reference_seconds) in zip(inputs, references):
        # Exclude first-use compilation from steady-state timings.
        warmup = model(ids, use_cache=False)
        del warmup
        torch.cuda.synchronize()
        start = perf_counter()
        logits = model(ids, use_cache=False).logits
        torch.cuda.synchronize()
        seconds = perf_counter() - start
        actual = logits.float().cpu()
        reference = reference.float()
        relative_error = ((actual - reference).norm() / reference.norm().clamp_min(1e-12)).item()
        targets = ids[:, 1:].cpu().unsqueeze(-1)
        actual_ll = actual[:, :-1].log_softmax(-1).gather(-1, targets).mean().item()
        reference_ll = reference[:, :-1].log_softmax(-1).gather(-1, targets).mean().item()
        ll_error = abs(actual_ll - reference_ll)
        print(f"Triton check shape={tuple(ids.shape)} logits_relative_error={relative_error:.6g} "
              f"mean_logprob_error={ll_error:.6g} pytorch_seconds={reference_seconds:.3f} "
              f"triton_seconds={seconds:.3f}", flush=True)
        if not torch.isfinite(actual).all() or relative_error > 0.005 or ll_error > 0.005:
            raise RuntimeError("Triton backend failed the packed-model numerical check")
        del logits
    return backend


def main(argv=None):
    args = parse_args(argv)
    output = result_path(args)
    if version("lm_eval") != LM_EVAL_VERSION:
        raise RuntimeError(f"Install lm-eval[hf]=={LM_EVAL_VERSION} for this evaluation")
    from lm_eval.utils import handle_non_serializable

    with ExitStack() as stack:
        log = None
        if output is not None:
            output.parent.mkdir(parents=True, exist_ok=True)
            log = stack.enter_context(output.with_suffix(".log").open("x"))
            stack.enter_context(redirect_stdout(_Tee(sys.stdout, log)))
            stack.enter_context(redirect_stderr(_Tee(sys.stderr, log)))
        start = perf_counter()
        try:
            torch.manual_seed(args.seed)
            print(f"Loading checkpoint: {args.model_path}", flush=True)
            model, tokenizer = load_model(args)
            print(f"Modeling implementation: {describe_model_impl(model)}", flush=True)
            results = run_lm_eval(
                model, tokenizer, tasks=args.tasks, batch_size=args.batch_size,
                num_fewshot=0, limit=args.limit, seed=args.seed,
                max_length=args.max_length, include_gsm8k=args.include_gsm8k,
                max_batch_size=args.max_batch_size,
            )
            elapsed = perf_counter() - start
            print(f"Evaluation complete in {elapsed:.2f} seconds", flush=True)
            if output is None:
                return
            results["experiment"] = {
                "model_path": str(args.model_path), "is_fp": args.is_fp,
                "model_class": describe_model_impl(model), "model_dtype": args.model_dtype,
                "device": args.device, "trust_remote_code": args.trust_remote_code,
                "backend": args.backend,
                "tasks": args.tasks, "include_gsm8k": args.include_gsm8k,
                "num_fewshot": 0, "apply_chat_template": False,
                "batch_size": args.batch_size, "max_length": args.max_length,
                "max_batch_size": args.max_batch_size,
                "seed": args.seed, "wall_seconds": elapsed,
                "versions": {name: version(name) for name in ("lm_eval", "torch", "transformers", "datasets", "hqq")},
            }
            with output.open("x") as stream:
                json.dump(results, stream, indent=2, default=handle_non_serializable)
                stream.write("\n")
            print(f"Results saved to: {output}", flush=True)
        except Exception:
            if log is not None:
                traceback.print_exc(file=log)
            raise


if __name__ == "__main__":
    main()
