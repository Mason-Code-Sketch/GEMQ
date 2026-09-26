"""
Loading helpers for quantized checkpoints.

`AutoHQQHFModel.from_quantized` loads the config and builds the model itself, dropping
`trust_remote_code`. For a checkpoint whose config.json carries an `auto_map` (DeepSeek-V2
ships modeling_deepseek.py with the weights) transformers then silently falls back to its
built-in implementation, which computes the same weights differently and scores worse.

`gemq/quantize.py` does load with trust_remote_code, so without this helper the
quantization-time perplexity and the real-quant numbers come from different code.
"""

import contextlib
import math
import os

import torch
import transformers


def ensure_deepseek_v2_remote_code_compat():
    """Provide a removed Transformers utility required by DeepSeek-V2's remote code."""
    from transformers.utils import import_utils

    if hasattr(import_utils, "is_torch_fx_available"):
        return

    # The official model only uses this on Torch versions older than 1.13.
    def is_torch_fx_available():
        return False

    import_utils.is_torch_fx_available = is_torch_fx_available


def _as_module_dir(path):
    """hqq passes `<checkpoint>/config.json`; resolving an auto_map needs the directory."""
    if not isinstance(path, (str, os.PathLike)):
        return path
    text = os.fspath(path)
    if os.path.basename(text) == "config.json" and os.path.isfile(text):
        return os.path.dirname(text) or "."
    return path


@contextlib.contextmanager
def force_remote_code():
    """Inject `trust_remote_code=True` into the auto-class calls hqq makes internally."""
    auto_model_classes = [transformers.AutoModelForCausalLM, transformers.AutoModel]

    original_from_pretrained = transformers.AutoConfig.from_pretrained
    originals_from_config = [cls.from_config for cls in auto_model_classes]

    def patched_from_pretrained(*args, **kwargs):
        kwargs.setdefault("trust_remote_code", True)
        if args:
            args = (_as_module_dir(args[0]),) + args[1:]
        elif "pretrained_model_name_or_path" in kwargs:
            kwargs["pretrained_model_name_or_path"] = _as_module_dir(
                kwargs["pretrained_model_name_or_path"]
            )
        return original_from_pretrained(*args, **kwargs)

    def make_patched_from_config(original):
        def patched_from_config(*args, **kwargs):
            kwargs.setdefault("trust_remote_code", True)
            return original(*args, **kwargs)
        return patched_from_config

    transformers.AutoConfig.from_pretrained = patched_from_pretrained
    for cls, original in zip(auto_model_classes, originals_from_config):
        cls.from_config = make_patched_from_config(original)

    try:
        yield
    finally:
        transformers.AutoConfig.from_pretrained = original_from_pretrained
        for cls, original in zip(auto_model_classes, originals_from_config):
            cls.from_config = original


def load_quantized_model(
    model_path, compute_dtype=torch.float16, device="cuda", trust_remote_code=True
):
    """Load a real-quant checkpoint, honouring the modeling code shipped inside it."""
    from hqq.models.hf.base import AutoHQQHFModel
    from gemq.inference.qwen2_moe import (
        QWEN2_HQQ_EXPERTS_CONFIG_KEY,
        Qwen2MoeHQQModel,
    )

    context = force_remote_code() if trust_remote_code else contextlib.nullcontext()
    with context:
        config = transformers.AutoConfig.from_pretrained(
            model_path, trust_remote_code=trust_remote_code
        )
        hqq_model_cls = (
            Qwen2MoeHQQModel
            if getattr(config, QWEN2_HQQ_EXPERTS_CONFIG_KEY, False)
            else AutoHQQHFModel
        )
        model = hqq_model_cls.from_quantized(
            model_path, compute_dtype=compute_dtype, device=device
        )
    return model


def describe_model_impl(model):
    """`transformers_modules.*` = the checkpoint's own code; `transformers.models.*` = built-in."""
    cls = type(model)
    return f"{cls.__module__}.{cls.__name__}"


def uses_remote_code(model):
    return type(model).__module__.startswith("transformers_modules")


def deepseek_mscale_correction(config):
    """
    The YaRN mscale HF's built-in DeepSeek-V2 omits from its attention scale. Pure; 1.0
    when the config has no YaRN.

    Official modeling_deepseek.py:
        softmax_scale = q_head_dim ** -0.5 * yarn_get_mscale(factor, mscale_all_dim) ** 2
    HF built-in:
        scaling = qk_head_dim ** -0.5          # mscale missing

    The cos/sin side already matches: transformers' _compute_yarn_parameters derives
    attention_factor from the same mscale/mscale_all_dim ratio the official code uses.
    """
    rope_scaling = getattr(config, "rope_scaling", None) or {}
    factor = rope_scaling.get("factor")
    mscale_all_dim = rope_scaling.get("mscale_all_dim")
    if not factor or not mscale_all_dim:
        return 1.0
    return (0.1 * mscale_all_dim * math.log(factor) + 1.0) ** 2


def align_deepseek_softmax_scale(model, verbose=True):
    """
    Apply `deepseek_mscale_correction` to every attention module. Returns the factor
    applied, 1.0 if nothing changed.

    No-op on the official implementation, and on transformers versions that already fix
    this (PR #47435, merged 2026-07-20, not in any release as of v5.14.1): a module whose
    `scaling` already differs from the plain qk_head_dim ** -0.5 baseline is left alone,
    so upgrading transformers cannot silently double-apply the factor.
    """
    if uses_remote_code(model):
        return 1.0

    correction = deepseek_mscale_correction(model.config)
    if correction == 1.0:
        return 1.0

    patched, already = 0, 0
    for module in model.modules():
        if not (hasattr(module, "qk_head_dim") and hasattr(module, "scaling")):
            continue
        if math.isclose(module.scaling, module.qk_head_dim ** -0.5, rel_tol=1e-6):
            module.scaling *= correction
            patched += 1
        else:
            already += 1

    if verbose and patched:
        print(f"Applied YaRN mscale^2={correction:.4f} to {patched} attention modules")
    if verbose and already:
        print(f"YaRN mscale already applied upstream; left {already} attention modules alone")
    return correction if patched else 1.0
