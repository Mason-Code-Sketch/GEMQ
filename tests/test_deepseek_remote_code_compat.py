from transformers.utils import import_utils

from gemq.utils.hf_loading import ensure_deepseek_v2_remote_code_compat


def test_restores_removed_deepseek_remote_code_import(monkeypatch):
    monkeypatch.delattr(import_utils, "is_torch_fx_available", raising=False)

    ensure_deepseek_v2_remote_code_compat()

    assert import_utils.is_torch_fx_available() is False
