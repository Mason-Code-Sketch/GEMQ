"""Resident, contiguous-layer placement for packed DeepSeek evaluation."""

import torch


def layer_device_map(model, devices):
    layers = model.model.layers
    if not devices or len(set(devices)) != len(devices) or len(devices) > len(layers):
        raise ValueError("Use distinct devices, with at least one decoder layer per device")
    mapping = {"model.embed_tokens": devices[0]}
    for index in range(len(layers)):
        mapping[f"model.layers.{index}"] = devices[index * len(devices) // len(layers)]
    mapping.update({"model.norm": devices[-1], "lm_head": devices[-1]})
    return mapping


@torch.no_grad()
def distribute_packed_model(model, devices):
    from accelerate import dispatch_model
    from hqq.core.quantize import HQQLinear

    mapping = layer_device_map(model, devices)
    for name, device in mapping.items():
        module = model.get_submodule(name)
        # HQQ metadata and packed codes need its own mover, not just Module.to().
        for child in module.modules():
            if isinstance(child, HQQLinear):
                child.cuda(device)
        module.to(device)
    return dispatch_model(model, device_map=mapping, main_device=devices[0])
