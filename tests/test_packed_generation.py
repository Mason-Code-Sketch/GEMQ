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




class PackedForwardTests(unittest.TestCase):
    @staticmethod
    def unpack(codes, levels, width):
        e, _, n = codes.shape
        ids = torch.stack((codes & 15, codes >> 4), dim=2).reshape(e, width, n).permute(0, 2, 1).long()
        return levels.permute(0, 3, 1, 2).gather(-1, ids.reshape(e, n, width // 128, 128)).reshape(e, n, width)

    @staticmethod
    def dense(inputs, weights, offs):
        output = inputs.new_empty((inputs.shape[0], weights.shape[-1]))
        start = 0
        for expert, end in enumerate(offs.tolist()):
            output[start:end] = inputs[start:end] @ weights[expert]
            start = end
        return output

    def exercise(self, tokens, top_k, fused, dtype, prefill=False):
        from unittest.mock import patch
        from copy import deepcopy
        from gemq.packed_generation import _bind_packed_forward, GroupedCachedExperts
        native = GroupedCachedExperts.forward
        module = GroupedCachedExperts.__new__(GroupedCachedExperts)
        torch.nn.Module.__init__(module)
        module.num_experts = top_k + 1
        module.compute_dtype = dtype
        module.fp32_topk = False
        module.act_fn = torch.nn.functional.silu
        torch.manual_seed(4)
        names = ('gate_up_proj', 'down_proj') if fused else ('gate_proj', 'up_proj', 'down_proj')
        for name in names:
            rows = 256 if name == 'gate_up_proj' else 128
            weight = (torch.randint(-3, 4, (module.num_experts, rows, 128)).to(dtype) / 128)
            module.register_buffer(name, weight)
        module.eval()
        reference = deepcopy(module)
        records = {}
        for name in names:
            weight = getattr(module, name)
            codes, levels = pack_weight(weight)
            placeholder = torch.empty(weight.shape, device='meta', dtype=dtype)
            setattr(module, name, placeholder)
            records[id(placeholder)] = (codes, levels, 128)
        stats = dict(decode_calls=0, prefill_calls=0)
        bounds = []
        def kernel(inputs, codes, levels, offsets, max_rows_per_expert):
            bounds.append(max_rows_per_expert)
            lengths = torch.diff(offsets, prepend=offsets.new_zeros(1))
            self.assertLessEqual(int(lengths.max()), max_rows_per_expert)
            self.assertEqual(max_rows_per_expert, tokens)
            return self.dense(inputs, self.unpack(codes, levels, 128).transpose(1, 2), offsets)
        _bind_packed_forward(module, records, stats, top_k, kernel, self.unpack)
        hidden = torch.randn(tokens, 128).to(dtype)
        # Every token selects the same experts; the final expert remains empty.
        indices = torch.arange(top_k).flip(0).repeat(tokens, 1)
        weights = torch.softmax(torch.randn(tokens, top_k), -1).to(dtype)
        with torch.no_grad(), patch.dict(native.__globals__, {'_linear': lambda x, w, o: self.dense(x, w.transpose(1, 2), o)}), patch.object(torch, '_grouped_mm', self.dense):
            expected = reference(hidden, indices, weights)
            actual = module(hidden, indices, weights)
        self.assertTrue(torch.equal(expected, actual))
        self.assertIs(GroupedCachedExperts.forward, native)
        self.assertEqual(stats['prefill_calls'], len(names) if prefill else 0)
        self.assertEqual(stats['decode_calls'], len(names) if tokens and not prefill else 0)
        self.assertEqual(len(bounds), stats['decode_calls'])
        with torch.no_grad(), self.assertRaisesRegex(ValueError, 'top-k'):
            module(hidden, indices[:, :-1], weights[:, :-1])

    def test_decode_bound_and_native_output(self):
        for top_k, tokens in ((2, 1), (2, 13), (2, 17), (8, 42), (8, 84)):
            for fused, dtype in ((True, torch.float16), (False, torch.float16),
                                 (True, torch.bfloat16), (False, torch.bfloat16)):
                with self.subTest(top_k=top_k, tokens=tokens, fused=fused):
                    self.exercise(tokens, top_k, fused, dtype)

    def test_invalid_router_contract_rejected_before_conversion(self):
        from gemq.packed_generation import GroupedCachedExperts
        module = GroupedCachedExperts.__new__(GroupedCachedExperts)
        torch.nn.Module.__init__(module)
        module.num_experts = 8
        module.fp32_topk = False
        model = torch.nn.Sequential(module).eval()
        for top_k in (None, 0, -1, True, 1.5, 9):
            model.config = SimpleNamespace(model_type='mixtral', num_experts_per_tok=top_k)
            with self.subTest(top_k=top_k), self.assertRaises(ValueError):
                prepare_packed_generation(SimpleNamespace(model=model))
            self.assertNotIn('forward', module.__dict__)
        module.fp32_topk = True
        model.config.num_experts_per_tok = 2
        with self.assertRaises(ValueError):
            prepare_packed_generation(SimpleNamespace(model=model))

    def test_prefill_boundary(self):
        self.exercise(256, 8, True, torch.float16)
        self.exercise(257, 8, False, torch.float16, prefill=True)


if __name__ == '__main__': unittest.main()
