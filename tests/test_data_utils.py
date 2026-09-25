from types import SimpleNamespace
from unittest.mock import patch

import random

import torch

from gemq.utils.data_utils import get_wikitext2


class _Split(list):
    def __getitem__(self, key):
        if key == "text":
            return [row["text"] for row in self]
        return super().__getitem__(key)


class _Tokenizer:
    def __init__(self):
        self.texts = []

    def __call__(self, text, **_kwargs):
        self.texts.append(text)
        return SimpleNamespace(input_ids=torch.arange(len(text) + 1).reshape(1, -1))


def test_wikitext2_calibration_matches_moe_ptq_corpus_windows():
    train = _Split([{"text": "abcdefgh"}, {"text": "ijklmnop"}])
    test = _Split([{"text": "qrst"}, {"text": "uvwx"}])
    tokenizer = _Tokenizer()

    with (
        patch("gemq.utils.data_utils._load_split", side_effect=[train, test]),
        patch("gemq.utils.data_utils.AutoTokenizer.from_pretrained", return_value=tokenizer),
    ):
        loader, testenc = get_wikitext2(
            nsamples=4,
            seed=42,
            seqlen=4,
            model="unused",
            dataset_root="unused",
        )

    train_tokens = torch.arange(len("abcdefgh\n\nijklmnop") + 1)
    starts = random.Random(42).sample(range(train_tokens.numel() - 4 + 1), 4)

    assert tokenizer.texts == ["abcdefgh\n\nijklmnop", "qrst\n\nuvwx"]
    assert torch.equal(testenc.input_ids[0], torch.arange(len("qrst\n\nuvwx") + 1))
    assert len(starts) == len(set(starts))
    for (input_ids, targets), start in zip(loader, starts):
        assert torch.equal(input_ids[0], train_tokens[start:start + 4])
        assert torch.equal(targets[:, :-1], torch.full((1, 3), -100))
        assert torch.equal(targets[:, -1], input_ids[:, -1])
