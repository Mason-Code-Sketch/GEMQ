"""Unit coverage for checkpoint-path validation in the quantization entry point."""

from types import SimpleNamespace
import unittest

from gemq.quantize import validate_checkpoint_paths


class CheckpointPathValidationTest(unittest.TestCase):
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
