import os
import argparse
import time
import gc
import json
import pickle
from functools import partial
from tqdm import tqdm

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, logging
from hqq.models.hf.base import AutoHQQHFModel

from gemq.quantizers.gptq import MCMoeGPTQWeightQuantizer, GPTQWeightQuantizer
from gemq.utils.data_utils import get_calib_loader
from gemq.utils.model_utils import *
from gemq.utils.quant_utils import *
from gemq.utils.eval_utils import evaluate_perplexity, run_lm_eval
from gemq.utils.hf_loading import align_deepseek_softmax_scale
from gemq.resource_ledger import ResourceLedger

logging.set_verbosity_error()


def make_gptq_quantizer(weight, name, wbits, args):
    hidden_size = weight.shape[1]
    if hidden_size % args.groupsize == 0:
        groupsize = args.groupsize
    else:
        assert hidden_size % 64 == 0, "Currently only supports groupsize=64 as fallback."
        groupsize = 64
        if args.verbose:
            print(f"Forcing groupsize from {args.groupsize} to 64 for module: {name}")

    quantizer_cls = (
        MCMoeGPTQWeightQuantizer if args.reproduce_mcmoe else GPTQWeightQuantizer
    )
    return quantizer_cls(
        weight,
        name,
        wbits,
        args.blocksize,
        args.percdamp,
        groupsize,
        args.actorder,
        args.static_groups,
        args.mse,
    )


def get_qwen2_expert_bits(args, allocation, layer_idx, num_routed_experts):
    num_scored_experts = num_routed_experts + 1
    if not args.mixed:
        return [args.expert_wbits] * num_scored_experts
    if allocation is None:
        raise ValueError("Mixed Qwen1.5 quantization requires a bit allocation.")
    layer_allocation = allocation[layer_idx]
    expected_expert_ids = set(range(num_scored_experts))
    actual_expert_ids = set(layer_allocation)
    if actual_expert_ids != expected_expert_ids:
        raise ValueError(
            f"Qwen1.5 layer {layer_idx} allocation has expert ids "
            f"{sorted(actual_expert_ids)}, expected {sorted(expected_expert_ids)}."
        )
    return [layer_allocation[expert_id] for expert_id in range(num_scored_experts)]


def build_qwen2_expert_quantizers(moe_block, expert_bits, args):
    num_routed_experts = get_qwen2_num_routed_experts(moe_block)
    if len(expert_bits) != num_routed_experts + 1:
        raise ValueError(
            "Qwen1.5 expert bit allocation must include every routed expert and "
            f"one shared expert, got {len(expert_bits)} entries for "
            f"{num_routed_experts} routed experts."
        )
    quantizers = {}
    for expert_id, bitwidth in enumerate(expert_bits[:-1]):
        if bitwidth >= 16:
            continue
        quantizers[expert_id] = {
            "gate_up_proj": make_gptq_quantizer(
                moe_block.experts.gate_up_proj[expert_id],
                f"mlp.experts.{expert_id}.gate_up_proj",
                bitwidth,
                args,
            ),
            "down_proj": make_gptq_quantizer(
                moe_block.experts.down_proj[expert_id],
                f"mlp.experts.{expert_id}.down_proj",
                bitwidth,
                args,
            ),
        }
    return quantizers


def collect_qwen2_expert_hessians(moe_block, inputs, quantizers):
    hidden_states = inputs[0].detach().reshape(-1, inputs[0].shape[-1])
    _, _, selected_experts = moe_block.gate(hidden_states)
    for expert_id, expert_quantizers in quantizers.items():
        token_indices = (selected_experts == expert_id).nonzero(as_tuple=True)[0]
        if token_indices.numel() == 0:
            continue
        expert_inputs = hidden_states[token_indices]
        expert_quantizers["gate_up_proj"].add_batch(expert_inputs)
        gate, up = F.linear(
            expert_inputs, moe_block.experts.gate_up_proj[expert_id]
        ).chunk(2, dim=-1)
        expert_activations = moe_block.experts.act_fn(gate) * up
        expert_quantizers["down_proj"].add_batch(expert_activations)


@torch.no_grad()
def quantize_qwen2_expert_weights(moe_block, quantizers, args):
    for expert_id, expert_quantizers in quantizers.items():
        for projection_name, quantizer in expert_quantizers.items():
            quantized, scales, zeros = quantizer.quantize()
            dequantized = quantizer.dequantize(quantized, scales, zeros)
            target = getattr(moe_block.experts, projection_name)[expert_id]
            target.copy_(
                dequantized.reshape_as(target).to(
                    device=target.device,
                    dtype=target.dtype,
                )
            )
            if args.verbose:
                print(
                    f"| mlp.experts.{expert_id}.{projection_name:<15} | "
                    f"{quantizer.nbits:<3} | {quantizer.groupsize:>4} | {'fused':>9} |"
                )


def save_quantized_model(model, tokenizer, save_path, save_dtype, real_quant):
    """
    Save the real/pseudo quantized model.
    """
    if real_quant:
        tokenizer.save_pretrained(save_path)
        AutoHQQHFModel.save_quantized(model, save_path)
    else:
        dtype = torch.float16 if save_dtype == "float16" else torch.bfloat16
        model = model.to(dtype)
        tokenizer.save_pretrained(save_path)
        model.save_pretrained(save_path)


def capture_router_finetune_state(model, router_params):
    """Capture identity-based router and frozen-parameter checks before AdamW."""
    router_ids = {id(param) for param in router_params}
    if not router_ids:
        raise RuntimeError("No router parameters were selected for fine-tuning.")
    if len(router_ids) != len(router_params):
        raise RuntimeError("Router parameter selection contains duplicates.")

    return {
        "router_ids": router_ids,
        "router_snapshots": {
            id(param): param.detach().clone() for param in router_params
        },
        # Optimizer in-place updates increment Tensor._version. Recording versions
        # avoids cloning every frozen parameter in a large MoE model.
        "frozen_versions": {
            id(param): param._version
            for param in model.parameters()
            if id(param) not in router_ids
        },
    }


def validate_router_finetune_state(model, router_params, state):
    """Verify that AdamW changed router parameters and no frozen parameter."""
    parameters_by_id = {id(param): param for param in model.parameters()}
    router_ids = state["router_ids"]
    if set(parameters_by_id).intersection(router_ids) != router_ids:
        raise RuntimeError("A selected router parameter is no longer present in the model.")

    changed_frozen = [
        parameter_id
        for parameter_id, version in state["frozen_versions"].items()
        if parameters_by_id[parameter_id]._version != version
    ]
    if changed_frozen:
        raise RuntimeError("Frozen parameters changed during router fine-tuning.")

    max_router_update = max(
        (
            (parameters_by_id[parameter_id].detach() - before)
            .abs()
            .max()
            .item()
            for parameter_id, before in state["router_snapshots"].items()
        ),
        default=0.0,
    )
    if max_router_update <= 0.0:
        raise RuntimeError("Routers did not change during fine-tuning.")
    return max_router_update


def finetune_routers(model, dataloader, args):
    """
    Fine-tune all router modules in the MoE model.
    """
    # disable kv cahce
    use_cache = model.config.use_cache
    model.config.use_cache = False
    org_dtype = next(model.parameters()).dtype

    model.train()
    model = model.to(torch.bfloat16)

    # prepare dataset
    input_ids = []
    for data in dataloader:
        input_ids.append(data[0])  # (1, seqlen)
    input_ids = torch.cat(input_ids, dim=0)  # (nsamples, seqlen)

    # enable gradients for all routers
    router_params = get_router_params(model, args.model_name)  # NOTE: return a list of parameters
    for p in model.parameters():
        p.requires_grad = False
    for p in router_params:
        p.requires_grad = True

    sanity_state = capture_router_finetune_state(model, router_params)
    if args.verbose:
        print("Router parameters selected for fine-tuning:", len(router_params))

    # start fine-tuning
    optimizer = torch.optim.AdamW(router_params, lr=args.rft_lr, weight_decay=args.rft_wd)
    for epoch in range(args.rft_epochs):
        loss_sum = 0.
        start = time.time()
        for i in range(args.nsamples // args.rft_batch_size):
            idx = i * args.rft_batch_size
            data = input_ids[idx: idx + args.rft_batch_size].to("cuda")  # (bsz, seqlen)
            outputs = model(input_ids=data, labels=data)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_sum += loss.item()
            if i % 32 == 0:
                print(f"[epoch {epoch} | iter {i:>3d}] loss: {loss_sum / (i+1):.6f}")
        elapse = time.time() - start
        print(f"epoch {epoch:>2} loss: {loss_sum / len(dataloader):.6f}, elapse: {elapse:.2f} seconds")

        # sanity check
        if epoch == 0:
            max_router_update = validate_router_finetune_state(
                model, router_params, sanity_state
            )
            print("Sanity check passed!")
            if args.verbose:
                print("Max router parameter update:", max_router_update)

    # restore
    model = model.to(org_dtype)
    model.config.use_cache = use_cache


@torch.no_grad()
def quantize_weights_gptq(model, dataloader, args):
    """
    Perform mixed-precision weight-only quantization with GPTQ quantizer.
    At the end of this function, the model weights are replaced with dequantized weights in fp16.

    NOTE: only supports single GPU quantization.
    """
    # disable KV cache for quantization
    use_cache = model.config.use_cache
    model.config.use_cache = False

    # build a bit allocation config for each Linear module
    bit_cfg = build_alloc_cfg(model, args)
    model_type = NAME_TO_MODEL[args.model_name]
    qwen2_allocation = None
    if model_type == ModelType.QWEN2MOE and args.mixed:
        with open(args.bit_cfg, "rb") as file:
            qwen2_allocation = pickle.load(file)

    # prepare decoder inputs and kwargs for model forward
    inps, layer_kwargs = compute_decoder_inputs(model, dataloader, args.model_name, "cuda")

    # retrieve decoder blocks
    layers = get_blocks(model, args.model_name)
    validate_qwen2_moe_model(model, args.model_name)

    # perform quantization for each block
    quant_modules = {}
    outs = torch.zeros_like(inps)
    for i in tqdm(range(len(layers)), desc="GPTQ Quantizing"):
        if args.verbose:
            print("+" + "="*57 + "+")
            print(f"| block {i:<24} | {'bit':<3} |  gs  | {'time (s)':>9} |")
            print("+" + "-"*57 + "+")
        start = time.time()

        # retrieve linear modules in the current block
        layer = layers[i].to("cuda")
        named_linears = get_named_linears(layer)
        moe_block = (
            get_moe_block(layer, args.model_name)
            if model_type == ModelType.QWEN2MOE else None
        )
        qwen2_num_routed_experts = (
            get_qwen2_num_routed_experts(moe_block)
            if moe_block is not None else None
        )

        # create a quantizer for each linear module that requires quantization
        quantizers = {}
        for name, m in named_linears.items():
            wbits = bit_cfg[i][name]

            # skip
            if wbits >= 16:
                continue

            # NOTE: adjust groupsize to fit the hidden size
            hidden_size = m.weight.shape[1]
            if hidden_size % args.groupsize == 0:
                groupsize = args.groupsize
            else:
                assert hidden_size % 64 == 0, "Currently only supports groupsize=64 as fallback."
                groupsize = 64
                if args.verbose:
                    print(f"Forcing groupsize from {args.groupsize} to 64 for module: {name}")

            if args.reproduce_mcmoe:
                quantizers[name] = MCMoeGPTQWeightQuantizer(
                    m.weight.data, name, wbits, args.blocksize, args.percdamp,
                    groupsize, args.actorder, args.static_groups, args.mse
                )
            else:
                quantizers[name] = GPTQWeightQuantizer(
                    m.weight.data, name, wbits, args.blocksize, args.percdamp,
                    groupsize, args.actorder, args.static_groups, args.mse
                )
            
            # collect quantized modules for real quantization saving
            quant_modules[f"{i}.{name}"] = m

        # update Hessian using a batch of input data for each linear module
        def update_hessian_hook(m, x, y, quantizer):
            x = x[0].detach()
            quantizer.add_batch(x)

        handles = []
        for name in named_linears:
            if name not in quantizers:
                continue
            handles.append(
                named_linears[name].register_forward_hook(
                    partial(update_hessian_hook, quantizer=quantizers[name])
                )
            )
        qwen2_quantizers = None
        qwen2_handle = None
        if model_type == ModelType.QWEN2MOE:
            qwen2_quantizers = build_qwen2_expert_quantizers(
                moe_block,
                get_qwen2_expert_bits(
                    args, qwen2_allocation, i, qwen2_num_routed_experts
                ),
                args,
            )
            qwen2_handle = moe_block.register_forward_pre_hook(
                partial(collect_qwen2_expert_hessians, quantizers=qwen2_quantizers)
            )
        for j in range(args.nsamples):
            batch_inps = inps[j: j+1]
            outs[j] = get_decoder_hidden_states(
                layer(batch_inps, **layer_kwargs), batch_inps.shape
            )
        for h in handles:
            h.remove()
        if qwen2_handle is not None:
            qwen2_handle.remove()

        elapse = time.time() - start
        if args.verbose:
            print(f"| {'pre-quantization':<30}   {' ':<3} | {' ':>4} | {elapse:>9.2f} |")
            print("+" + "-"*57 + "+")

        # quantize each linear module
        for name, m in named_linears.items():
            if name not in quantizers:
                continue

            # quantize
            start = time.time()
            Q, scales, zeros = quantizers[name].quantize()

            # dequantize
            W = quantizers[name].dequantize(Q, scales, zeros)

            # replace weights and register quantization params
            m.weight.data = W.reshape_as(m.weight.data)
            m.register_buffer("quant_scales", scales)
            m.register_buffer("quant_zeros", zeros)
            m.register_buffer("quant_nbits", torch.tensor(quantizers[name].nbits))
            m.register_buffer("quant_groupsize", torch.tensor(quantizers[name].groupsize))

            elapse = time.time() - start
            if args.verbose:
                print(f"| {name:<30} | {quantizers[name].nbits:<3} | {quantizers[name].groupsize:>4} | {elapse:>9.2f} |")

        if qwen2_quantizers is not None:
            quantize_qwen2_expert_weights(moe_block, qwen2_quantizers, args)

        # compute layer outputs using quantized weights
        start = time.time()

        for j in range(args.nsamples):
            batch_inps = inps[j: j+1]
            outs[j] = get_decoder_hidden_states(
                layer(batch_inps, **layer_kwargs), batch_inps.shape
            )

        elapse = time.time() - start
        if args.verbose:
            print("+" + "-"*57 + "+")
            print(f"| {'post-quantization':<30}   {' ':<3} | {' ':>4} | {elapse:>9.2f} |")
            print("+" + "="*57 + "+")

        # update inputs for the next layer
        inps, outs = outs, inps

        # save memory
        layers[i] = layer.to("cpu")
        del quantizers
        gc.collect()
        torch.cuda.empty_cache()

    model.config.use_cache = use_cache  # restore

    return quant_modules


def parse_args():
    parser = argparse.ArgumentParser(description="GEMQ for MoE-LLMs Quantization.")
    parser.add_argument(
        "--verbose", action="store_true",
        help="Whether to enable verbose logging"
    )
    
    # model args
    parser.add_argument(
        "--model", type=str, required=True,
        help="Path to the pre-trained model or shortcut name",
    )
    parser.add_argument(
        "--model_name", type=str, required=True,
        help="Name of the model; used to load model-specific modules",
    )
    parser.add_argument(
        "--model_dtype", type=str, default="float16", choices=["auto", "float16", "bfloat16"],
        help="Data type of the model weights; use `auto` to load the model in the default dtype",
    )
    parser.add_argument(
        "--use_fast", action="store_true",
        help="Whether to use the fast tokenizer implementation",
    )
    parser.add_argument(
        "--attn_impl", type=str, default="eager", choices=["eager", "sdpa"],
        help="Implementation of attention to use",
    )
    parser.add_argument(
        "--disable_cache", action="store_true",
        help="Disable KV cache",
    )
    parser.add_argument(
        "--trust_remote_code", action="store_true",
        help="Enable `trust_remote_code` when loading the model from HuggingFace Hub",
    )

    # dataset args
    parser.add_argument(
        "--calib_dataset", type=str, default="wikitext2",
        help="Which calibration dataset to use",
    )
    parser.add_argument(
        "--dataset_root", type=str, default=None,
        help="Optional directory containing local c4_gptq_new_seed0 and wikitext2 DatasetDicts",
    )
    parser.add_argument(
        "--nsamples", type=int, default=128,
        help="Number of calibration sequences"
    )
    parser.add_argument(
        "--seqlen", type=int, default=2048,
        help="Length of each sequence",
    )
    parser.add_argument(
        "--batch_size", type=int, default=1,
        help="Batch size of the data loader"
    )
    parser.add_argument(
        "--seed", type=int, default=0,
        help="Seed for sampling the calibration data"
    )

    # mixed-precision bit allocation
    parser.add_argument(
        "--mixed", action="store_true",
        help="Whether to mixed-precision quantization"
    )
    parser.add_argument(
        "--bit_cfg", type=str, default="",
        help="Path to the bit allocation config file; leave blank to use uniform allocation"
    )

    # quantization args
    parser.add_argument(
        "--eval_fp", action="store_true",
        help="Whether to skip quantization and evaluate the full-precision model"
    )
    parser.add_argument(
        "--quantizer", type=str, default="gptq",
        help="Which quantizer to use"
    )
    parser.add_argument(
        "--attn_wbits", type=int, default=4,
        help="#bits for quantization of attention modules"
    )
    parser.add_argument(
        "--gate_wbits", type=int, default=16,
        help="#bits for quantization of gate (router) modules"
    )
    parser.add_argument(
        "--dense_wbits", type=int, default=4,
        help="#bits for quantization of dense modules"
    )
    parser.add_argument(
        "--expert_wbits", type=int, default=4,
        help="#bits for quantization of expert modules"
    )
    parser.add_argument(
        "--groupsize", type=int, default=128,
        help="Groupsize to use for quantization"
    )
    
    # quantizer-specific args
    parser.add_argument(
        "--blocksize", type=int, default=128,
        help="Blocksize to use for quantization"
    )
    parser.add_argument(
        "--percdamp", type=float, default=0.01,
        help="Percent of the average Hessian diagonal to use for dampening"
    )
    parser.add_argument(
        "--mse", action="store_true",
        help="Whether to seach for quantization parameters (range)"
    )
    parser.add_argument(
        "--actorder", action="store_true",
        help="Whether to apply the activation order GPTQ heuristic (never used)"
    )
    parser.add_argument(
        "--static_groups", action="store_true",
        help="Whether to use static groups; recommended when using `--actorder` for more efficient inference. (never used)"
    )
    parser.add_argument(
        "--reproduce_mcmoe", action="store_true",
        help="Whether to use the GPTQ implementation from MC-MoE"
    )

    # router fine-tuning args
    parser.add_argument(
        "--finetune_routers", action="store_true",
        help="Whether to finetune the router modules after quantization"
    )
    parser.add_argument(
        "--rft_epochs", type=int, default=1,
        help="Number of epochs for the router fine-tuning"
    )
    parser.add_argument(
        "--rft_batch_size", type=int, default=1,
        help="Batch size for the router fine-tuning"
    )
    parser.add_argument(
        "--rft_lr", type=float, default=0.0001,
        help="Learning rate for the router fine-tuning"
    )
    parser.add_argument(
        "--rft_wd", type=float, default=0.0001,
        help="Weight decay for the router fine-tuning"
    )

    # evaluation args
    parser.add_argument(
        "--eval_downstream", action="store_true",
        help="Whether to run evaluation on downstream tasks"
    )
    parser.add_argument(
        "--downstream_tasks", type=str, default="piqa,arc_easy,arc_challenge,hellaswag,winogrande,mathqa,mmlu",
        help="Tasks to evaluate on; ignored if `--eval_downstream` is False"
    )
    parser.add_argument(
        "--lm_eval_batchsize", type=int, default=32,
        help="Batch size for lm_eval downstream evaluation"
    )
    parser.add_argument(
        "--num_fewshot", type=int, default=0,
        help="Few-shot examples to use for downstream evaluation"
    )

    # i/o args
    parser.add_argument(
        "--real_quant", action="store_true",
        help="Whether to conduct real quantization and save the int weights (using HQQ)"
    )
    parser.add_argument(
        "--save_path", type=str, default="",
        help="Save quantized checkpoint under this path"
    )
    parser.add_argument(
        "--save_dtype", type=str, default="float16", choices=["float16", "bfloat16"],
        help="Data type to save the quantized model"
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

    if args.real_quant and NAME_TO_MODEL[args.model_name] == ModelType.QWEN2MOE:
        raise ValueError("Qwen1.5-MoE-A2.7B supports fake quantization only.")

    with resource_ledger.command():
        with resource_ledger.component("load_model_and_tokenizer"):
            print("Loading model ...")
            tokenizer = AutoTokenizer.from_pretrained(
                args.model,
                use_fast=args.use_fast,
                trust_remote_code=args.trust_remote_code,
            )
            model = AutoModelForCausalLM.from_pretrained(
                args.model,
                device_map="cpu",
                torch_dtype=args.model_dtype,
                attn_implementation=args.attn_impl,
                trust_remote_code=args.trust_remote_code,
            )
            align_deepseek_softmax_scale(model)
            model.seqlen = 2048
            model.eval()

        with resource_ledger.component("load_calibration"):
            print("Loading calibration data ...")
            dataloader = get_calib_loader(tokenizer, args)

        quant_modules = {}
        if not args.eval_fp:
            print("Start quantizing model weights ...")
            quantizer = args.quantizer.lower().split("-")[0]
            if quantizer != "gptq":
                raise ValueError(f"Unsupported weight quantizer: {args.quantizer}")
            with resource_ledger.component("quant_gptq"):
                quant_modules = quantize_weights_gptq(model, dataloader, args)

        if args.finetune_routers:
            with resource_ledger.component("prepare_router_ft"):
                model = dispatch_model_to_all_devices(model)

            with resource_ledger.component("evaluation_gptq"):
                print("Evaluating quantized model before fine-tuning ...")
                evaluate_perplexity(
                    model,
                    tokenizer,
                    ["wikitext2", "c4"],
                    args.model_name,
                    offload=False,
                    dataset_root=args.dataset_root,
                )

            with resource_ledger.component("ft_routers"):
                print("Fine-tuning routers ...")
                finetune_routers(model, dataloader, args)

        print("Evaluating model ...")
        model.eval()
        if args.eval_downstream or args.finetune_routers:
            if not args.finetune_routers:
                with resource_ledger.component("prepare_evaluation"):
                    model = dispatch_model_to_all_devices(model)

            with resource_ledger.component("evaluation_router_ft"):
                evaluate_perplexity(
                    model,
                    tokenizer,
                    ["wikitext2", "c4"],
                    args.model_name,
                    offload=False,
                    dataset_root=args.dataset_root,
                )
                if args.eval_downstream:
                    if args.disable_cache:
                        model.config.use_cache = False
                    try:
                        run_lm_eval(
                            model,
                            tokenizer,
                            tasks=args.downstream_tasks.split(","),
                            batch_size=args.lm_eval_batchsize,
                            num_fewshot=args.num_fewshot,
                        )
                    except Exception:
                        print("Downstream evaluation failed. Skipping ...")
        else:
            with resource_ledger.component("evaluation"):
                evaluate_perplexity(
                    model,
                    tokenizer,
                    ["wikitext2", "c4"],
                    args.model_name,
                    offload=True,
                    dataset_root=args.dataset_root,
                )

        if args.save_path:
            with resource_ledger.component("checkpoint_router_ft"):
                print("Saving model ...")
                os.makedirs(args.save_path, exist_ok=True)

                if args.real_quant:
                    replace_linears(
                        model,
                        args.model_name,
                        quant_modules,
                        quant_weight=True,
                    )
                    check_packing(model, quant_modules, args)
                else:
                    for name, module in quant_modules.items():
                        module.quant_scales = None
                        module.quant_zeros = None
                        module.quant_nbits = None
                        module.quant_groupsize = None

                save_quantized_model(
                    model,
                    tokenizer,
                    args.save_path,
                    args.save_dtype,
                    args.real_quant,
                )
                print("Quantized model saved to:", args.save_path)
