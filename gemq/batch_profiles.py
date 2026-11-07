"""Model-specific auto-batch caps; explicit caller values always take precedence."""

DEFAULT_CAPS = {"seven": 51, "mmlu_5shot": 17, "gsm8k_0shot": 96}
MIXTRAL_CAPS = {"seven": 34, "mmlu_5shot": 13, "gsm8k_0shot": 48}
DEEPSEEK_CAPS = {"seven": 8, "mmlu_5shot": 8, "gsm8k_0shot": 16}
QWEN15_CAPS = {"seven": 12, "mmlu_5shot": 12, "gsm8k_0shot": 24}
QWEN3_CAPS = {"seven": 51, "mmlu_5shot": 60, "gsm8k_0shot": 96}
MODEL_CAPS = {
    "qwen3_moe": QWEN3_CAPS,
    "mixtral": MIXTRAL_CAPS,
    "deepseek_v2": DEEPSEEK_CAPS,
    "qwen2_moe": QWEN15_CAPS,
}


def auto_batch_cap(model, profile, explicit=None):
    profile = "seven" if profile == "zeroshot" else profile
    if explicit is not None:
        return explicit
    model_type = getattr(getattr(model, "config", None), "model_type", None)
    caps = MODEL_CAPS.get(model_type, DEFAULT_CAPS)
    return caps[profile]


def resolve_cli_batch_cap(args, model, profile):
    # Parsed legacy defaults are provisional until the actual model is loaded.
    explicit = getattr(args, "max_batch_size", None)
    if not getattr(args, "_max_batch_size_explicit", True):
        explicit = None
    args.max_batch_size = auto_batch_cap(model, profile, explicit)
    return args.max_batch_size
