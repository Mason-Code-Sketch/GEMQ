"""Exercise GPTQ grid storage across the actual HQQ checkpoint boundary."""
import torch
from hqq.models.hf.base import AutoHQQHFModel
from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM

from gemq.inference.fused_moe import hqq_state_from_quantized_weight, replace_fused_moe_experts
from gemq.quantizers.gptq import GPTQWeightQuantizer
from gemq import eval_zeroshot
from gemq.utils.quant_utils import create_hqq_linear_from_quantized_weights


def test_fp32_grid_checkpoint_round_trip(tmp_path, monkeypatch):
    torch.manual_seed(11)
    config = Qwen3MoeConfig(vocab_size=32, hidden_size=16, num_hidden_layers=1,
                            intermediate_size=32, moe_intermediate_size=16,
                            num_attention_heads=2, num_key_value_heads=2, head_dim=8,
                            num_experts=4, num_experts_per_tok=2)
    config.gemq_fp32_grid = True
    model = Qwen3MoeForCausalLM(config).to(torch.bfloat16).eval()
    attn = model.model.layers[0].self_attn
    weight = attn.q_proj.weight.detach()
    solver = GPTQWeightQuantizer(weight, nbits=3, groupsize=8, blocksize=8)
    solver.add_batch(torch.randn(32, 16).bfloat16())
    codes, scale, zero = solver.quantize()
    expected = solver.dequantize(codes, scale, zero).reshape_as(weight).bfloat16()
    assert not torch.equal(scale, scale.bfloat16().float())
    attn.q_proj = create_hqq_linear_from_quantized_weights(
        codes, scale, zero, weight.shape, 3, 8, device="cpu")
    experts = model.model.layers[0].mlp.experts
    states = {}
    for expert in range(config.num_experts):
        states[expert] = {}
        for name in ("gate_up_proj", "down_proj"):
            matrix = getattr(experts, name)[expert].detach()
            quantizer = GPTQWeightQuantizer(matrix, nbits=3, groupsize=8, blocksize=8)
            quantizer.add_batch(torch.randn(32, matrix.shape[1]).bfloat16())
            q, scales, zeros = quantizer.quantize()
            matrix.copy_(quantizer.dequantize(q, scales, zeros).reshape_as(matrix))
            states[expert][name] = hqq_state_from_quantized_weight(
                q, scales, zeros, tuple(matrix.shape), 3, 8, matrix.device)
    assert replace_fused_moe_experts(model, {0: states}) == 0
    AutoHQQHFModel.save_quantized(model, str(tmp_path))
    monkeypatch.setattr(eval_zeroshot.AutoTokenizer, "from_pretrained", lambda *args, **kwargs: None)
    args = eval_zeroshot.parse_args(["--model_path", str(tmp_path), "--device", "cpu", "--backend", "pytorch"])
    restored, _ = eval_zeroshot.load_model(args)
    assert args.model_dtype == "bfloat16"
    projection = restored.model.layers[0].self_attn.q_proj
    for key, original in (("scale", scale), ("zero", zero)):
        torch.testing.assert_close(projection.meta[key], original, rtol=0, atol=0)
    torch.testing.assert_close(projection.dequantize(), expected, rtol=0, atol=0)
    with torch.no_grad():
        sample = torch.arange(6)[None, :]
        torch.testing.assert_close(restored.eval()(sample).logits, model(sample).logits, rtol=0, atol=0)
