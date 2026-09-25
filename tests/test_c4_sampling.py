import random
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from gemq.utils.data_utils import build_calib_loader, sample_gptq_c4_calibration


class _Split(list):
    pass


class _Tokenizer:
    model_max_length = 32

    def __call__(self, text, **_kwargs):
        return SimpleNamespace(
            input_ids=torch.tensor([[ord(character) for character in text]])
        )


def _moe_ptq_reference(tokenizer, dataset, num_samples, seq_len, seed):
    rng = random.Random(seed)
    spans, selections = [], []
    while len(spans) < num_samples:
        document_index = rng.randint(0, len(dataset) - 1)
        token_ids = tokenizer(dataset[document_index]["text"], return_tensors="pt").input_ids[0]
        if token_ids.numel() < seq_len:
            continue
        start = rng.randint(0, token_ids.numel() - seq_len)
        spans.append(token_ids[start:start + seq_len])
        selections.append({"document_index": document_index, "start": start})
    return torch.stack(spans), selections


class C4SamplingTest(unittest.TestCase):
    def setUp(self):
        self.dataset = _Split(
            [
                {"text": "ab"},
                {"text": "abcdefgh"},
                {"text": "ijklmnop"},
            ]
        )
        self.tokenizer = _Tokenizer()

    def test_sampling_matches_moe_ptq_reference(self):
        actual_spans, actual_selections = sample_gptq_c4_calibration(
            self.tokenizer,
            self.dataset,
            num_samples=6,
            seq_len=4,
            seed=17,
        )
        expected_spans, expected_selections = _moe_ptq_reference(
            self.tokenizer,
            self.dataset,
            num_samples=6,
            seq_len=4,
            seed=17,
        )

        self.assertTrue(torch.equal(actual_spans, expected_spans))
        self.assertEqual(actual_selections, expected_selections)

    def test_sampling_includes_final_legal_start(self):
        class _EndpointRandom:
            def __init__(self, _seed):
                self.values = iter([1, 4])

            def randint(self, lower, upper):
                value = next(self.values)
                if value != upper or value < lower:
                    raise AssertionError(
                        f"expected inclusive upper bound {upper}, got {value}"
                    )
                return value

        with patch("gemq.utils.data_utils.random.Random", _EndpointRandom):
            spans, selections = sample_gptq_c4_calibration(
                self.tokenizer,
                self.dataset,
                num_samples=1,
                seq_len=4,
                seed=0,
            )

        self.assertEqual(selections, [{"document_index": 1, "start": 4}])
        self.assertTrue(torch.equal(spans[0], torch.tensor([101, 102, 103, 104])))

    def test_c4_stats_loader_uses_random_windows(self):
        expected, _ = _moe_ptq_reference(
            self.tokenizer,
            self.dataset,
            num_samples=5,
            seq_len=4,
            seed=23,
        )
        with patch("gemq.utils.data_utils._load_split", return_value=self.dataset):
            loader = build_calib_loader(
                "c4",
                self.tokenizer,
                max_block_size=4,
                n_blocks_for_stat=5,
                batch_size=2,
                num_workers=0,
                seed=23,
                dataset_root="unused",
            )

        actual = torch.cat([batch["input_ids"] for batch in loader])
        self.assertTrue(torch.equal(actual, expected))


if __name__ == "__main__":
    unittest.main()
