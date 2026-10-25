"""Unit coverage for checkpoint-path validation in the quantization entry point."""

from types import SimpleNamespace
import unittest
from unittest.mock import patch

from gemq.quantize import evaluate_unpacked_perplexity, validate_checkpoint_paths


class CheckpointPathValidationTest(unittest.TestCase):
    def test_final_packed_only_requires_a_saved_evaluated_packed_model(self):
        fields = dict(final_packed_eval_only=True, real_quant=True,
                      eval_real_quant=True, save_path="packed", fake_save_path="",
                      eval_downstream=False)
        validate_checkpoint_paths(SimpleNamespace(**fields))
        for key, value in (("real_quant", False), ("eval_real_quant", False),
                           ("save_path", ""), ("eval_downstream", True)):
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_checkpoint_paths(SimpleNamespace(**(fields | {key: value})))

    def test_final_packed_only_skips_unpacked_ppl_without_changing_default(self):
        with patch("gemq.quantize.evaluate_perplexity", return_value="result") as evaluate:
            args = SimpleNamespace(final_packed_eval_only=True)
            self.assertIsNone(evaluate_unpacked_perplexity(
                "model", "tokenizer", ["c4"], "name", args, offload=False))
            evaluate.assert_not_called()
            args.final_packed_eval_only = False
            self.assertEqual(evaluate_unpacked_perplexity(
                "model", "tokenizer", ["c4"], "name", args, offload=False), "result")
            evaluate.assert_called_once_with(
                "model", "tokenizer", ["c4"], "name", offload=False)

    def test_fake_checkpoint_path_requires_a_distinct_real_checkpoint(self):
        validate_checkpoint_paths(
            SimpleNamespace(
                eval_real_quant=True,
                real_quant=True,
                fake_save_path="fake",
                save_path="real",
            )
        )

        with self.assertRaisesRegex(ValueError, "requires --real_quant"):
            validate_checkpoint_paths(
                SimpleNamespace(
                    eval_real_quant=False,
                    real_quant=False,
                    fake_save_path="fake",
                    save_path="real",
                )
            )

        with self.assertRaisesRegex(ValueError, "must differ"):
            validate_checkpoint_paths(
                SimpleNamespace(
                    eval_real_quant=False,
                    real_quant=True,
                    fake_save_path="checkpoint",
                    save_path="checkpoint",
                )
            )
