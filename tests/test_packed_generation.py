import unittest
from types import SimpleNamespace
import torch
from gemq.packed_generation import pack_weight, prepare_packed_generation

class PackedGenerationTests(unittest.TestCase):
    def test_exact_round_trip_both_dtypes(self):
        for dtype in (torch.float16, torch.bfloat16):
            torch.manual_seed(12)
            # Independent grids across experts, rows and K groups.
            lut = torch.randn(2, 5, 3, 8).to(dtype)
            lut[..., 0] = 0.0
            lut[..., 1] = -0.0
            ids = torch.randint(8, (2, 5, 3, 128))
            w = lut.gather(-1, ids).reshape(2, 5, 384)
            q, table = pack_weight(w)
            got_ids = torch.stack((q & 15, q >> 4), dim=2).reshape(2, 384, 5).permute(0, 2, 1).long()
            tables = table.permute(0, 3, 1, 2)
            decoded = tables.gather(-1, got_ids.reshape(2, 5, 3, 128)).reshape_as(w)
            self.assertTrue(torch.equal(w.view(torch.int16), decoded.view(torch.int16)))
            self.assertEqual(table.dtype, dtype)
    def test_rejects_nonrepresentable_weights(self):
        for dtype in (torch.float16, torch.bfloat16):
            with self.assertRaises(ValueError):
                pack_weight(torch.arange(128).to(dtype).reshape(1, 1, 128))
    def test_rejects_wrong_layout(self):
        for w in (torch.zeros(1, 2, 128), torch.zeros(2,128,dtype=torch.float16), torch.zeros(1,2,127,dtype=torch.float16)):
            with self.assertRaises(ValueError): pack_weight(w)
    def test_other_model_unmodified(self):
        model = torch.nn.Linear(2, 2).eval()
        model.config = SimpleNamespace(model_type='qwen2_moe')
        before = model.weight.clone()
        self.assertFalse(prepare_packed_generation(SimpleNamespace(model=model))['enabled'])
        self.assertTrue(torch.equal(model.weight,before))
    def test_training_unmodified(self):
        model = torch.nn.Linear(2,2)
        model.config = SimpleNamespace(model_type='mixtral')
        self.assertFalse(prepare_packed_generation(SimpleNamespace(model=model))['enabled'])

if __name__ == '__main__': unittest.main()
