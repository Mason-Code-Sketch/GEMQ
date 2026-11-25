from gemq.quantizers.rtn import normal_quantize

import pytest
import torch


@pytest.mark.parametrize("bits,mcmoe,expected", [
    (1, False, [2.578125, -2.578125, 2.578125, 2.578125, -2.578125, .58203125, .58203125, .58203125]),
    (2, False, [-.9609375, -.9609375, -.9609375, 2.640625, -8.1875, 1.25, .416015625, 0]),
    (2, True, [-.95703125, -.95703125, -.95703125, 2.625, -8.125, 1.28125, .427734375, 0]),
    (3, False, [-.4453125, -1.9921875, 1.09375, 2.640625, -8.1875, 1.234375, .53125, 0]),
    (3, True, [-.4453125, -1.984375, 1.09375, 2.640625, -8.125, 1.234375, .52734375, 0]),
])
def test_bf16_candidates_match_kdmoq_with_short_tail(bits, mcmoe, expected):
    weight = torch.tensor([[0, -1.25, .375, 2.75, -8.5, 1.25, .5, 0], [0] * 8], dtype=torch.bfloat16)
    actual = normal_quantize(weight, blocksize=5, wbit=bits, mcmoe_rules=mcmoe)
    torch.testing.assert_close(actual, weight.new_tensor([expected, [0] * 8]), rtol=0, atol=0)
