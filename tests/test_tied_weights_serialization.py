import pytest
import torch
from transformers import PretrainedConfig, PreTrainedModel

from gemq.utils.hf_loading import normalize_legacy_tied_weights_for_serialization


class _LegacyRemoteCodeConfig(PretrainedConfig):
    model_type = "legacy-remote-code-test"

    def __init__(self, tie_word_embeddings=False, **kwargs):
        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)


class _LegacyRemoteCodeModel(PreTrainedModel):
    config_class = _LegacyRemoteCodeConfig
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.proj = torch.nn.Linear(2, 2)
        self.post_init()


class _LegacyTiedRemoteCodeModule:
    def __init__(self):
        self.config = type("Config", (), {"tie_word_embeddings": True})()
        self._tied_weights_keys = ["lm_head.weight"]

    def modules(self):
        yield self


def test_normalizes_untied_legacy_tied_weight_list_for_serialization(tmp_path):
    model = _LegacyRemoteCodeModel(_LegacyRemoteCodeConfig())

    normalize_legacy_tied_weights_for_serialization(model)
    model.save_pretrained(tmp_path)

    assert model._tied_weights_keys == {}
    assert (tmp_path / "config.json").is_file()


def test_rejects_legacy_list_when_weights_are_tied():
    model = _LegacyTiedRemoteCodeModule()

    with pytest.raises(ValueError, match="cannot be serialized safely"):
        normalize_legacy_tied_weights_for_serialization(model)
