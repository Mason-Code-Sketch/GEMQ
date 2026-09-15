"""Model identifiers supported by GEMQ without importing model implementations."""

from enum import Enum, auto


class ModelType(Enum):
    LLAMA2 = auto()
    QWEN3 = auto()
    MIXTRAL = auto()
    DEEPSEEKV2 = auto()
    OLMOE = auto()
    QWEN3MOE = auto()
    QWEN2MOE = auto()


NAME_TO_MODEL = {
    "meta-llama/Llama-2-7b-hf": ModelType.LLAMA2,
    "Qwen/Qwen3-8B": ModelType.QWEN3,
    "mistralai/Mixtral-8x7B-v0.1": ModelType.MIXTRAL,
    "Mixtral-8x7B-v0.1": ModelType.MIXTRAL,
    "deepseek-ai/DeepSeek-V2-Lite": ModelType.DEEPSEEKV2,
    "DeepSeek-V2-Lite": ModelType.DEEPSEEKV2,
    "allenai/OLMoE-1B-7B-0924": ModelType.OLMOE,
    "Qwen/Qwen3-30B-A3B": ModelType.QWEN3MOE,
    "Qwen3-30B-A3B": ModelType.QWEN3MOE,
    "Qwen3-30B-A3B-Base": ModelType.QWEN3MOE,
    "Qwen/Qwen1.5-MoE-A2.7B": ModelType.QWEN2MOE,
    "Qwen1.5-MoE-A2.7B": ModelType.QWEN2MOE,
}
