"""Unit coverage for checkpoint-path validation in the quantization entry point."""

from types import ModuleType, SimpleNamespace
import unittest
import sys
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

from gemq.quantize import save_quantized_model, validate_checkpoint_paths


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


class PackedCheckpointCodeTest(unittest.TestCase):
    def test_remote_model_code_is_saved_and_loadable(self):
        from transformers.dynamic_module_utils import get_class_from_dynamic_module

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            module_file = source / "modeling_checkpoint.py"
            module_file.write_text("class TinyModel:\n    marker = 17\n")
            module_name = "transformers_modules.checkpoint_save.modeling_checkpoint"
            module = ModuleType(module_name)
            module.__file__ = str(module_file)
            exec(module_file.read_text(), module.__dict__)
            target = root / "packed"
            target.mkdir()
            tokenizer = Mock()
            with patch.dict(sys.modules, {module_name: module}), patch(
                "gemq.quantize.AutoHQQHFModel.save_quantized"
            ) as save:
                save_quantized_model(module.TinyModel(), tokenizer, str(target), "float16", True)
            save.assert_called_once()
            tokenizer.save_pretrained.assert_called_once_with(str(target))
            self.assertEqual((target / module_file.name).read_bytes(), module_file.read_bytes())
            restored = get_class_from_dynamic_module(
                "modeling_checkpoint.TinyModel", str(target), local_files_only=True
            )
            self.assertEqual(restored.marker, 17)

    def test_native_model_does_not_copy_transformers_sources(self):
        model = Mock()
        with patch("gemq.quantize.AutoHQQHFModel.save_quantized") as save, patch(
            "gemq.quantize.custom_object_save"
        ) as copy_code:
            save_quantized_model(model, Mock(), "packed", "float16", True)
        save.assert_called_once_with(model, "packed")
        copy_code.assert_not_called()
