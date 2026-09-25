import os
import os.path as osp
import argparse
import pickle
import json
import time
import gc
from collections import defaultdict
from functools import partial
from tqdm import tqdm

import torch
import torch.nn as nn

from transformers import AutoTokenizer, AutoModelForCausalLM, logging

from gemq.utils.data_utils import get_calib_loader
from gemq.utils.model_utils import *
from gemq.quantizers.rtn import MCMoeRTNWeightQuantizer
from gemq.utils.hf_loading import align_deepseek_softmax_scale
from gemq.resource_ledger import ResourceLedger

logging.set_verbosity_error()


def get_inout_hook(m, x, y, inps, outs):
    # save inputs and outputs
    if isinstance(x, (list, tuple)):
        inps.append(x[0])  # (bsz, seqlen, hidden_size)
    else:
        inps.append(x)  # (bsz, seqlen, hidden_size)

    if isinstance(y, (list, tuple)):
        outs.append(y[0])  # (bsz, seqlen, hidden_size)
    else:
        outs.append(y)  # (bsz, seqlen, hidden_size)


@torch.inference_mode()
def get_stats(model, enc, args):
    """
    Forward to compute router statistics and quantization (reconstruction) loss of each layer.
    """
    model.config.use_cache = False
    num_batches = enc.shape[0]

    # get model-specific info
    model_name = args.model_name
    model_type = NAME_TO_MODEL[model_name]
    gate_stats_hook_fn = get_gate_stats_hook_fn(model_name)
    sublinear_names = get_sublinear_names(model_name)  # e.g., ["gate_proj", "up_proj", "down_proj"]

    # retrieve blocks that require quantization
    layers = get_blocks(model, model_name)
    validate_qwen2_moe_model(model, model_name)

    # get input and kwargs to the first layer decoding layer
    inps = []
    layer_kwargs = {}

    move_embed(model, model_name, "cuda")
    layers[0] = layers[0].to("cuda")
    
    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps.append(inp)  # NOTE: inp is (bsz, seqlen, hidden_size)
            layer_kwargs.update(kwargs)
            raise ValueError  # early exit to break later inference

    layers[0] = Catcher(layers[0])
    for i in range(num_batches):
        batch = enc[i].to("cuda")  # (bsz, seqlen)
        try:
            model(batch)
        except ValueError:
            pass
    layers[0] = layers[0].module  # restore
    inps = torch.stack(inps, dim=0)  # (num_batches, bsz, seqlen, hidden_size)
    
    # for memory savings
    move_embed(model, model_name, "cpu")
    layers[0] = layers[0].cpu()
    gc.collect()
    torch.cuda.empty_cache()

    # forward layer-by-layer to collect stats
    outs = torch.zeros_like(inps)
    act_weights, act_counts, quant_loss = {}, {}, {}
    for i in tqdm(range(len(layers)), desc="Computing stats"):
        layer = layers[i].to("cuda")

        # NOTE: we skip computing stats for the first dense layer of deepseekv2
        if model_type == ModelType.DEEPSEEKV2 and i == 0:
            
            # still need to forward it to get inputs for the next layer
            for j in range(num_batches):
                batch_inps = inps[j]
                outs[j] = get_decoder_hidden_states(
                    layer(batch_inps, **layer_kwargs), batch_inps.shape
                )
            layers[i] = layer.to("cpu")
            gc.collect()
            torch.cuda.empty_cache()
            inps, outs = outs, inps
            continue
        

        named_linears = get_named_linears(layer)
        moe_block = get_moe_block(layer, model_name)

        # get expert weights & counts and inputs/outputs of the moe block
        block_inps, block_outs = [], []
        _weights, _counts = [], []
        # register hook
        handle = moe_block.register_forward_hook(
            partial(gate_stats_hook_fn, inps=block_inps, outs=block_outs, weights=_weights, counts=_counts)
        )
        for j in range(num_batches):
            batch_inps = inps[j]
            outs[j] = get_decoder_hidden_states(
                layer(batch_inps, **layer_kwargs), batch_inps.shape
            )
        act_weights[i] = sum(_weights)  # (num_routed_experts + 1 if has_shared_expert else 0,)
        act_counts[i] = sum(_counts)    # (num_routed_experts + 1 if has_shared_expert else 0,)
        # remove hook
        handle.remove()

        if model_type == ModelType.QWEN2MOE:
            bit_cfg = list(map(int, args.wbits.split(",")))
            quant_loss[i] = compute_qwen2_moe_quantization_losses(
                moe_block, block_inps, block_outs, bit_cfg
            )
            layers[i] = layer.to("cpu")
            gc.collect()
            torch.cuda.empty_cache()
            inps, outs = outs, inps
            continue

        # compute expert quantization errors
        bit_cfg = list(map(int, args.wbits.split(",")))  # e.g., [1, 2, 3]
        expert_names = get_all_expert_names(model_name)  # NOTE: shared expert always at the end

        # register a quantizer for each expert linear at each bitwidth
        quantizers = {}
        for e, expert_name in enumerate(expert_names):
            quantizers[e] = {}
            for b in bit_cfg:
                quantizers[e][b] = {}
                for l, lname in enumerate(sublinear_names):
                    m = named_linears[f"{expert_name}.{lname}"]
                    quantizers[e][b][l] = MCMoeRTNWeightQuantizer(m.weight.data, nbits=b)

        # compute reconstruction loss of block output caused by quantization (i.e., perturbation)
        layer_quant_loss = defaultdict(dict)
        for e, expert_name in enumerate(expert_names):
            # cache unquantized weights
            if "shared" in expert_name:
                org_sd = get_shared_expert_block(moe_block).state_dict()
            else:
                org_sd = moe_block.experts[e].state_dict()

            # for each bitwidth
            for b in bit_cfg:
                # quantize all linear modules in the expert in-place
                for l, lname in enumerate(sublinear_names):
                    m = named_linears[f"{expert_name}.{lname}"]
                    m.weight.data = quantizers[e][b][l].quantize()
                
                # compute diff
                loss = 0
                for j in range(num_batches):
                    quant_outs = moe_block(block_inps[j])[0]  # (bsz, seqlen, hidden_size)
                    loss += torch.norm(block_outs[j].double() - quant_outs.double()).item()
                layer_quant_loss[e][b] = loss

                # restore unquantized weights
                if "shared" in expert_name:
                    get_shared_expert_block(moe_block).load_state_dict(org_sd)
                else:
                    moe_block.experts[e].load_state_dict(org_sd)

        quant_loss[i] = layer_quant_loss


        layers[i] = layer.to("cpu")
        gc.collect()
        torch.cuda.empty_cache()
        inps, outs = outs, inps

    return act_weights, act_counts, quant_loss


@torch.inference_mode()
def compute_mcmoe_stats(model, dataloader, args):
    """
    Compute MC-MoE statistics, including expert activation weights, activation counts,
    and quantization (reconstruction) loss of each expert from each layer.

    The results will be saved to `args.mcmoe_stats_dir` as pickle files.
    """
    # convert dataloader format
    enc = []
    for data in dataloader:
        enc.append(data[0])  # (bsz, seqlen)
    enc = torch.stack(enc, dim=0)  # (num_batch, bsz, seqlen)

    # forward to get stats
    #   act_weights: {layer_idx: Tensor [num_experts,]}
    #   act_counts:  {layer_idx: Tensor [num_experts,]}
    #   quant_loss: {layer_idx: {expert_idx: {bitwidth: loss}}}
    act_weights, act_counts, quant_loss = get_stats(model, enc, args)

    # save results
    os.makedirs(args.mcmoe_stats_dir, exist_ok=True)
    with open(osp.join(args.mcmoe_stats_dir, "experts_act_weights.pkl"), "wb") as f:
        pickle.dump(act_weights, f)
    with open(osp.join(args.mcmoe_stats_dir, "experts_act_counts.pkl"), "wb") as f:
        pickle.dump(act_counts, f)
    with open(osp.join(args.mcmoe_stats_dir, "experts_quant_loss.pkl"), "wb") as f:
        pickle.dump(quant_loss, f)
    print("Model stats saved to:", args.mcmoe_stats_dir)


def compute_layer_grads(model, dataloader, args):
    """
    Compute outputs gradients wrt to the task loss of all decoder layers.
    """
    model.config.use_cache = False

    # NOTE: disable aux loss
    model.config.alpha = 0.0

    # register hooks to get activation gradients
    layer_output_grads = defaultdict(list)

    def get_gradient_hook(m, grad_input, grad_output, grads):
        # grad_output is a tuple; we're only interested in the first element
        grads.append(grad_output[0].cpu())  # (bsz, seqlen, hidden_size)

    layers = get_blocks(model, args.model_name)
    handles = []
    for i in range(len(layers)):
        handle = layers[i].register_full_backward_hook(
            partial(get_gradient_hook, grads=layer_output_grads[i])
        )
        handles.append(handle)

    # accumulate gradients
    model.zero_grad()
    for data in tqdm(dataloader, desc="Computing gradients"):
        x = data[0].cuda()
        outputs = model(input_ids=x, labels=x)
        loss = outputs.loss
        loss.backward()

    # remove hooks
    for handle in handles:
        handle.remove()

    # combine results of each layer
    for i in range(len(layers)):
        layer_output_grads[i] = torch.stack(layer_output_grads[i], dim=0)  # (num_batches, bsz, seqlen, hidden_size)
    
    # save gradients
    os.makedirs(osp.dirname(args.layer_grads_path), exist_ok=True)
    print(f"Saving layer output gradients to: {args.layer_grads_path} ... ", end="")
    start = time.time()
    torch.save(layer_output_grads, args.layer_grads_path)
    print(f"Done in {(time.time() - start)/60:.2f} minutes")
    print("Layer output gradients saved to:", args.layer_grads_path)


@torch.inference_mode()
def compute_qwen2_moe_layer_reconstruction_errors(
    moe_block,
    block_inps,
    block_outs,
    layer_sq_grads,
    bit_cfg,
    forward_batch_size,
):
    """Score Qwen1.5 routed experts stored in fused parameter tensors."""
    layer_quant_loss = defaultdict(dict)
    num_routed_experts = get_qwen2_num_routed_experts(moe_block)

    for expert_id in range(num_routed_experts + 1):
        if expert_id == num_routed_experts:
            modules = [
                moe_block.shared_expert.gate_proj,
                moe_block.shared_expert.up_proj,
                moe_block.shared_expert.down_proj,
            ]
            original_weights = [module.weight.detach().clone() for module in modules]
        else:
            original_weights = [
                moe_block.experts.gate_up_proj[expert_id].detach().clone(),
                moe_block.experts.down_proj[expert_id].detach().clone(),
            ]

        for bitwidth in bit_cfg:
            if expert_id == num_routed_experts:
                quantized_weights = [
                    MCMoeRTNWeightQuantizer(weight, nbits=bitwidth).quantize()
                    for weight in original_weights
                ]
                for module, quantized_weight in zip(modules, quantized_weights):
                    module.weight.copy_(quantized_weight)
            else:
                quantized_gate_up = MCMoeRTNWeightQuantizer(
                    original_weights[0], nbits=bitwidth
                ).quantize()
                quantized_down = MCMoeRTNWeightQuantizer(
                    original_weights[1], nbits=bitwidth
                ).quantize()
                moe_block.experts.gate_up_proj[expert_id].copy_(quantized_gate_up)
                moe_block.experts.down_proj[expert_id].copy_(quantized_down)

            loss = 0.0
            for start in range(0, block_inps.shape[0], forward_batch_size):
                end = start + forward_batch_size
                quant_block_outs = moe_block(block_inps[start:end])
                loss += (
                    layer_sq_grads[start:end]
                    * (block_outs[start:end].double() - quant_block_outs.double()).pow(2)
                ).sum().item()
            layer_quant_loss[expert_id][bitwidth] = loss

            if expert_id == num_routed_experts:
                for module, original_weight in zip(modules, original_weights):
                    module.weight.copy_(original_weight)
            else:
                moe_block.experts.gate_up_proj[expert_id].copy_(original_weights[0])
                moe_block.experts.down_proj[expert_id].copy_(original_weights[1])

    return layer_quant_loss


@torch.inference_mode()
def compute_qwen2_moe_quantization_losses(
    moe_block, block_inps, block_outs, bit_cfg
):
    """Compute MC-MoE-style output losses for Qwen1.5 fused experts."""
    layer_quant_loss = defaultdict(dict)
    num_routed_experts = get_qwen2_num_routed_experts(moe_block)

    for expert_id in range(num_routed_experts + 1):
        if expert_id == num_routed_experts:
            modules = [
                moe_block.shared_expert.gate_proj,
                moe_block.shared_expert.up_proj,
                moe_block.shared_expert.down_proj,
            ]
            original_weights = [module.weight.detach().clone() for module in modules]
        else:
            original_weights = [
                moe_block.experts.gate_up_proj[expert_id].detach().clone(),
                moe_block.experts.down_proj[expert_id].detach().clone(),
            ]

        for bitwidth in bit_cfg:
            if expert_id == num_routed_experts:
                quantized_weights = [
                    MCMoeRTNWeightQuantizer(weight, nbits=bitwidth).quantize()
                    for weight in original_weights
                ]
                for module, quantized_weight in zip(modules, quantized_weights):
                    module.weight.copy_(quantized_weight)
            else:
                moe_block.experts.gate_up_proj[expert_id].copy_(
                    MCMoeRTNWeightQuantizer(
                        original_weights[0], nbits=bitwidth
                    ).quantize()
                )
                moe_block.experts.down_proj[expert_id].copy_(
                    MCMoeRTNWeightQuantizer(
                        original_weights[1], nbits=bitwidth
                    ).quantize()
                )

            loss = 0.0
            for inputs, outputs in zip(block_inps, block_outs):
                quant_outputs = moe_block(inputs)
                loss += torch.norm(outputs.double() - quant_outputs.double()).item()
            layer_quant_loss[expert_id][bitwidth] = loss

            if expert_id == num_routed_experts:
                for module, original_weight in zip(modules, original_weights):
                    module.weight.copy_(original_weight)
            else:
                moe_block.experts.gate_up_proj[expert_id].copy_(original_weights[0])
                moe_block.experts.down_proj[expert_id].copy_(original_weights[1])

    return layer_quant_loss


@torch.inference_mode()
def compute_faster_layer_re(model, dataloader, args):
    """
    Compute layer reconstruction errors (perturbations) caused by quantization of
    each expert from that layer, weighted by the squared gradients of the layer
    outputs wrt the task loss.
    """
    model.config.use_cache = False

    # unify data format
    # NOTE: dataloader is a list of (input_ids, None) with input_ids of shape (bsz, seqlen)
    enc = []
    for data in dataloader:
        enc.append(data[0])  # (bsz, seqlen)
    enc = torch.stack(enc, dim=0)  # (num_samples, 1, seqlen) NOTE: assuming batchsize=1
    num_samples, _, _ = enc.shape

    # load layer output gradients from disk if available
    assert osp.exists(args.layer_grads_path), \
        f"Layer output gradients not found at: {args.layer_grads_path}. Please compute them first using `compute_layer_grads`."
    print(f"Loading layer output gradients from: {args.layer_grads_path} ... ", end="", flush=True)
    start = time.time()
    # NOTE: this might take a while for large models like Mixtral-8x7B
    layer_output_grads = torch.load(args.layer_grads_path, weights_only=False, map_location="cpu")
    print(f"Done in {(time.time() - start)/60:.2f} minutes")
    # {0: (nsamples, 1, seqlen, hidden_size), 1: ...}
    assert num_samples == layer_output_grads[0].shape[0], \
        "Mismatch between dataloader and layer output gradients. Check if the gradients are computed on the same dataset and under the same batch size."


    # get model-specific info
    model_name = args.model_name
    model_type = NAME_TO_MODEL[model_name]
    sublinear_names = get_sublinear_names(model_name)  # e.g., ["gate_proj", "up_proj", "down_proj"]
    fwd_bsz = args.forward_batch_size

    # retrieve decoder blocks
    layers = get_blocks(model, model_name)
    validate_qwen2_moe_model(model, model_name)

    # get input and kwargs to the first layer decoding layer
    inps = []
    layer_kwargs = {}

    move_embed(model, model_name, "cuda")
    layers[0] = layers[0].to("cuda")
    
    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps.append(inp)  # NOTE: inp is (bsz, seqlen, hidden_size)
            layer_kwargs.update(kwargs)
            raise ValueError  # early exit to break later inference

    layers[0] = Catcher(layers[0])
    for i in range(num_samples // fwd_bsz):
        batch = enc[i * fwd_bsz:(i + 1) * fwd_bsz, 0].to("cuda")  # (bsz, seqlen)
        try:
            model(batch)
        except ValueError:
            pass
    layers[0] = layers[0].module  # restore
    inps = torch.cat(inps, dim=0)  # (num_samples, seqlen, hidden_size)
    
    # for memory savings
    move_embed(model, model_name, "cpu")
    layers[0] = layers[0].cpu()
    gc.collect()
    torch.cuda.empty_cache()


    # forward layer-by-layer to collect stats
    quant_loss = {}
    outs = torch.zeros_like(inps)
    for i in tqdm(range(len(layers)), desc="Computing rec errors"):
        layer = layers[i].to("cuda")

        # NOTE: we skip computing stats for the first dense layer of deepseekv2
        if model_type == ModelType.DEEPSEEKV2 and i == 0:
            # still need to forward it to get inputs for the next layer
            for j in range(num_samples // fwd_bsz):
                batch_inps = inps[j * fwd_bsz:(j + 1) * fwd_bsz]
                outs[j * fwd_bsz:(j + 1) * fwd_bsz] = get_decoder_hidden_states(
                    layer(batch_inps, **layer_kwargs), batch_inps.shape
                )
            layers[i] = layer.to("cpu")
            gc.collect()
            torch.cuda.empty_cache()
            inps, outs = outs, inps
            continue


        named_linears = get_named_linears(layer)
        moe_block = get_moe_block(layer, model_name)

        # get unquantized layer outputs and moe block in/outs
        block_inps, block_outs = [], []
        handle = moe_block.register_forward_hook(partial(get_inout_hook, inps=block_inps, outs=block_outs))
        for j in range(num_samples // fwd_bsz):
            batch_inps = inps[j * fwd_bsz:(j + 1) * fwd_bsz]
            outs[j * fwd_bsz:(j + 1) * fwd_bsz] = get_decoder_hidden_states(
                layer(batch_inps, **layer_kwargs), batch_inps.shape
            )
        block_inps = torch.cat(block_inps, dim=0)  # (num_samples, seqlen, hidden_size)
        block_outs = torch.cat(block_outs, dim=0)  # (num_samples, seqlen, hidden_size)
        handle.remove()

        if model_type == ModelType.QWEN2MOE:
            bit_cfg = list(map(int, args.wbits.split(",")))
            layer_sq_grads = layer_output_grads[i].squeeze(1).double().pow(2).to("cuda")
            quant_loss[i] = compute_qwen2_moe_layer_reconstruction_errors(
                moe_block,
                block_inps,
                block_outs,
                layer_sq_grads,
                bit_cfg,
                fwd_bsz,
            )
            inps, outs = outs, inps
            layers[i] = layer.to("cpu")
            gc.collect()
            torch.cuda.empty_cache()
            continue

        # compute expert quantization errors
        bit_cfg = list(map(int, args.wbits.split(",")))  # e.g., [1, 2, 3]
        expert_names = get_all_expert_names(model_name)  # NOTE: shared expert always at the end

        # register a quantizer for each expert linear at each bitwidth
        quantizers = {}
        for e, expert_name in enumerate(expert_names):
            quantizers[e] = {}
            for b in bit_cfg:
                quantizers[e][b] = {}
                for l, lname in enumerate(sublinear_names):
                    m = named_linears[f"{expert_name}.{lname}"]
                    quantizers[e][b][l] = MCMoeRTNWeightQuantizer(m.weight.data, nbits=b)

        # compute reconstruction errors of block output caused by quantization (perturbation)
        layer_sq_grads = layer_output_grads[i].squeeze(1).double().pow(2).to("cuda")  # (nsamples, seqlen, hidden_size)
        layer_quant_loss = defaultdict(dict)
        for e, expert_name in enumerate(expert_names):
            # cache unquantized weights
            if "shared" in expert_name:
                org_sd = get_shared_expert_block(moe_block, model_name).state_dict()
            else:
                org_sd = moe_block.experts[e].state_dict()

            # for each bitwidth
            for b in bit_cfg:
                # quantize the whole expert in-place
                for l, lname in enumerate(sublinear_names):
                    m = named_linears[f"{expert_name}.{lname}"]
                    m.weight.data = quantizers[e][b][l].quantize()

                # compute output changes (weighted sum squared errors)
                loss = 0
                for j in range(num_samples // fwd_bsz):
                    weights = layer_sq_grads[j * fwd_bsz:(j + 1) * fwd_bsz]
                    quant_block_outs = moe_block(block_inps[j * fwd_bsz:(j + 1) * fwd_bsz])  # (bsz, seqlen, hidden_size)
                    # NOTE: for model that outputs a tuple
                    if isinstance(quant_block_outs, (list, tuple)):
                        quant_block_outs = quant_block_outs[0]
                    loss += (weights * (block_outs[j * fwd_bsz:(j + 1) * fwd_bsz].double() - quant_block_outs.double()).pow(2)).sum().item()
                layer_quant_loss[e][b] = loss

                # restore unquantized weights
                if "shared" in expert_name:
                    get_shared_expert_block(moe_block, model_name).load_state_dict(org_sd)
                else:
                    moe_block.experts[e].load_state_dict(org_sd)

        # store layer results
        quant_loss[i] = layer_quant_loss

        # update buffers
        inps, outs = outs, inps

        # save memory
        layers[i] = layer.to("cpu")
        gc.collect()
        torch.cuda.empty_cache()
        

    # save results
    os.makedirs(osp.dirname(args.layer_re_path), exist_ok=True)
    with open(args.layer_re_path, "wb") as f:
        pickle.dump(quant_loss, f)
    print("Weighted reconstruction errors saved to:", args.layer_re_path)



def parse_args():
    parser = argparse.ArgumentParser(description="Compute statistics of MoE models")
    parser.add_argument(
        "--mode", type=str, required=True, choices=["mcmoe_stats", "layer_grads", "layer_re"],
        help="Which statistics to compute: \n"
             "`mcmoe_stats`: compute expert activation weights, counts, and quantization loss;\n"
             "`layer_grads`: compute gradients of layer outputs w.r.t. final CE loss;\n"
             "`layer_re`: compute layer reconstruction errors weighted by layer output gradients."
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
        "--model_dtype", type=str, default="float16", choices=["float16", "bfloat16"],
        help="Data type of the model weights",
    )
    parser.add_argument(
        "--attn_impl", type=str, default="eager", choices=["eager", "sdpa"],
        help="Implementation of attention to use",
    )
    parser.add_argument(
        "--use_fast", action="store_true",
        help="Whether to use the fast tokenizer implementation",
    )
    
    # dataset args
    parser.add_argument(
        "--calib_dataset", type=str, default="c4",
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
        help="Batch size of the data loader (MC-MoE uses 8 by default)"
    )
    parser.add_argument(
        "--seed", type=int, default=0,
        help="Seed for sampling the calibration data"
    )
    parser.add_argument(
        "--forward_batch_size", type=int, default=1,
        help="Batch size for model forward pass (used in computing layer reconstruction errors)"
    )

    # quantization args
    parser.add_argument(
        "--wbits", type=str, default="1,2,3",
        help="#bits for computing expert quantization error (a string separated by ,)"
    )
    parser.add_argument(
        "--blocksize", type=int, default=128,
        help="Blocksize to use for quantization"
    )

    # misc args
    parser.add_argument(
        "--mcmoe_stats_dir", type=str, default="",
        help="Path to save the model statistics computed in `mcmoe_stats` mode"
    )
    parser.add_argument(
        "--layer_grads_path",  type=str, default="",
        help="Path to the layer activation gradients"
    )
    parser.add_argument(
        "--layer_re_path",  type=str, default="",
        help="Path to the weighted reconstruction errors"
    )
    parser.add_argument(
        "--resource_output", type=str, default="",
        help="Optional JSON path for wall-time and GPU-memory accounting",
    )

    return parser.parse_args()


def main(args) -> None:
    resource_ledger = ResourceLedger(args.resource_output)
    with resource_ledger.command():
        with resource_ledger.component("load_model_and_tokenizer"):
            tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=args.use_fast)
            model = AutoModelForCausalLM.from_pretrained(
                args.model,
                device_map=("auto" if args.mode == "layer_grads" else "cpu"),
                torch_dtype=args.model_dtype,
                attn_implementation=args.attn_impl,
                trust_remote_code=True,
            )
            align_deepseek_softmax_scale(model)
            model.seqlen = args.seqlen
            validate_qwen2_moe_model(model, args.model_name)
            if args.mode == "layer_grads":
                model.train()
            else:
                model.eval()

        with resource_ledger.component("load_calibration"):
            dataloader = get_calib_loader(tokenizer, args)

        if args.mode == "layer_grads":
            with resource_ledger.component("comp_grad"):
                compute_layer_grads(model, dataloader, args)
        elif args.mode == "layer_re":
            print("Using faster implementation that batches expert forwards.")
            with resource_ledger.component("comp_stats"):
                compute_faster_layer_re(model, dataloader, args)
        elif args.mode == "mcmoe_stats":
            with resource_ledger.component("comp_stats"):
                compute_mcmoe_stats(model, dataloader, args)


if __name__ == "__main__":
    args = parse_args()
    print(json.dumps(vars(args), indent=4))
    main(args)
