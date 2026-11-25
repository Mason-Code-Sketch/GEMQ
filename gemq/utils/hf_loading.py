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

import psutil

import torch
import transformers




def layer_grads_max_memory(gpu_weight_budget_gib):
    """Keep overflow weights in CPU memory instead of implicit disk offload."""
    if gpu_weight_budget_gib is None:
        return None
    if gpu_weight_budget_gib <= 0:
        raise ValueError("GPU weight budget must be positive")
    return {0: gpu_weight_budget_gib * 1024**3,
            "cpu": psutil.virtual_memory().available}


@contextlib.contextmanager
def sequential_weight_loading():
    """Bound load-time materialization without changing weights or placement."""
    key = "HF_DEACTIVATE_ASYNC_LOAD"
    previous = os.environ.get(key)
    os.environ[key] = "1"
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = previous


def ensure_deepseek_v2_remote_code_compat():
    """Provide a removed Transformers utility required by DeepSeek-V2's remote code."""
    from transformers.utils import import_utils

    if hasattr(import_utils, "is_torch_fx_available"):
        return

    # The official model only uses this on Torch versions older than 1.13.
    def is_torch_fx_available():
        return False

    import_utils.is_torch_fx_available = is_torch_fx_available


def install_deepseek_generation_cache_compat(model):
    """Adapt only legacy DeepSeek generation's unbounded dynamic-cache calls.

    Cache tensors and their update implementation remain owned by Transformers.
    Refresh seen_tokens before every prepare call; layer-local usable lengths
    come from the cache, so decode never reuses layer zero's updated length.
    """
    from types import MethodType
    from weakref import ref
    from transformers.cache_utils import DynamicCache

    if (getattr(model.config, "model_type", None) != "deepseek_v2"
            or not type(model).__module__.startswith("transformers_modules.")
            or getattr(model, "_gemq_legacy_cache_compat", False)):
        return False
    original = model.prepare_inputs_for_generation

    def prepare(self, input_ids, past_key_values=None, **kwargs):
        cache = past_key_values
        if isinstance(cache, DynamicCache):
            if any(getattr(cache, "is_sliding", [])) or getattr(cache, "offloading", False):
                raise ValueError("Legacy DeepSeek requires an unbounded resident DynamicCache")
            # Old remote modeling code queries these removed APIs before forward.
            if not hasattr(type(cache), "seen_tokens"):
                cache.seen_tokens = cache.get_seq_length()
            cache.get_max_length = lambda: None
            if not hasattr(cache, "get_usable_length"):
                cache_ref = ref(cache)
                cache.get_usable_length = (
                    lambda new_seq_length, layer_idx=0: cache_ref().get_seq_length(layer_idx))
        prepared = original(input_ids, past_key_values=past_key_values, **kwargs)
        # New GenerationMixin supplies position_ids for the full prefix. Legacy
        # DeepSeek only slices IDs it creates itself, leaving provided IDs long.
        if isinstance(prepared, dict) and cache is not None:
            positions = prepared.get("position_ids")
            tokens = prepared.get("input_ids")
            if positions is not None and tokens is not None:
                prepared["position_ids"] = positions[:, -tokens.shape[1]:]
        return prepared

    model.prepare_inputs_for_generation = MethodType(prepare, model)
    model._gemq_legacy_cache_compat = True
    return True


def normalize_legacy_tied_weights_for_serialization(model):
    """Adapt untied remote-code models to Transformers 5 serialization."""
    for module in model.modules():
        tied_weight_keys = getattr(module, "_tied_weights_keys", None)
        if not isinstance(tied_weight_keys, list):
            continue

        if getattr(getattr(module, "config", None), "tie_word_embeddings", False):
            raise ValueError(
                "Transformers 5 requires a mapping for tied _tied_weights_keys; "
                "the legacy list cannot be serialized safely."
            )
        module._tied_weights_keys = {}


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


def native_grid_dtype(config):
    """Storage precision for GEMQ checkpoints that reconstruct weights in FP32."""
    if getattr(config, "gemq_grid_storage_dtype", None) == "float16":
        return torch.float16
    if getattr(config, "gemq_fp32_grid", False) is True:
        return torch.float32
    return None


def _load_grid_model(loader, path, dtype, device, grid_dtype):
    """Use HQQ's model layout and saved state with the recorded grid precision."""
    from gemq.inference.fused_moe import hqq_linear_from_state

    model = loader.create_model(path, {})
    model.save_dir = path
    loader.setup_model(model)
    weights = loader.load_weights(path, device)

    def restore(module, params=None):
        state = weights.get(module.name)
        if state is None:
            return module.to(device=device, dtype=dtype)
        if "W_q" in state:
            return hqq_linear_from_state(state, device, dtype, grid_dtype=grid_dtype)
        for name, value in state.items():
            target = value.to(device=device, dtype=dtype if value.is_floating_point() else value.dtype)
            setattr(module, name, torch.nn.Parameter(target, requires_grad=False))
        return module

    loader.patch_model(model, restore, restore, dict.fromkeys(model.linear_tags))
    loader.post_module_load(model, weights)
    model.hqq_quantized = True
    model.base_class = loader
    return model


def load_quantized_model(
    model_path, compute_dtype=None, device="cuda", trust_remote_code=True
):
    """Load a real-quant checkpoint, honouring the modeling code shipped inside it."""
    from hqq.models.hf.base import AutoHQQHFModel
    from gemq.inference.fused_moe import (
        QWEN2_HQQ_EXPERTS_CONFIG_KEY,
        FUSED_HQQ_EXPERTS_CONFIG_KEY,
        FusedMoeHQQModel,
    )

    context = force_remote_code() if trust_remote_code else contextlib.nullcontext()
    with context:
        config = transformers.AutoConfig.from_pretrained(
            model_path, trust_remote_code=trust_remote_code
        )
        if getattr(config, "model_type", None) == "deepseek_v2":
            ensure_deepseek_v2_remote_code_compat()
        hqq_model_cls = (
            FusedMoeHQQModel
            if (getattr(config, QWEN2_HQQ_EXPERTS_CONFIG_KEY, False)
                or getattr(config, FUSED_HQQ_EXPERTS_CONFIG_KEY, False))
            else AutoHQQHFModel
        )
        grid_dtype = native_grid_dtype(config)
        if grid_dtype is not None:
            dtype = compute_dtype or getattr(config, "dtype", None) or torch.bfloat16
            model = _load_grid_model(hqq_model_cls, model_path, dtype, device, grid_dtype)
        else:
            model = hqq_model_cls.from_quantized(
                model_path, compute_dtype=compute_dtype or torch.float16, device=device
            )
    install_deepseek_generation_cache_compat(model)
    return model


def describe_model_impl(model):
    """`transformers_modules.*` = the checkpoint's own code; `transformers.models.*` = built-in."""
    cls = type(model)
    return f"{cls.__module__}.{cls.__name__}"


def uses_remote_code(model):
    return type(model).__module__.startswith("transformers_modules")


def deepseek_mscale_correction(config):
    """
    The YaRN factor required by DeepSeek-V2's attention scale. Pure; 1.0
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

    No-op on the official implementation and on Transformers versions that already
    include the YaRN correction: a module whose
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
