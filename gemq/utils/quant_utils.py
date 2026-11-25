import pickle

from hqq.core.quantize import HQQLinear, BaseQuantizeConfig
from hqq.core.quantize import Quantizer

from gemq.utils.model_utils import *
from gemq.utils.hf_loading import native_grid_dtype


def build_alloc_cfg(model, args):
    """
    Build a bit allocation config for the model. The config includes bitwidth for
    each Linear modules required quantization.

    Returns:
        bit_cfg: a list containing alloction cfg for Linear modules in each block/layer
        e.g.,
        [
            {"module1": 1, "module2": 3, "module3": 2},  # layer1 cfg
            ...  
        ]
    """
    # load bit allocation config
    # NOTE: current cfg only supports mixed-precision for expert-ffn
    if args.mixed:
        with open(args.bit_cfg, "rb") as f:
            expert_bit_cfg = pickle.load(f)

    layers = get_blocks(model, args.model_name)

    bit_cfg = []
    for i in range(len(layers)):
        layer = layers[i]
        named_linears = get_named_linears(layer)

        layer_bit_cfg = {}
        for name, m in named_linears.items():
            mtype = get_module_type(name, args.model_name)
            if mtype == LinearModuleType.ATTN:
                layer_bit_cfg[name] = args.attn_wbits
            elif mtype == LinearModuleType.GATE:
                layer_bit_cfg[name] = args.gate_wbits
            elif mtype == LinearModuleType.DENSE:
                layer_bit_cfg[name] = args.dense_wbits
            elif mtype == LinearModuleType.EXPERT:
                if args.mixed:
                    expert_id = get_expert_id(name, args.model_name)
                    layer_bit_cfg[name] = expert_bit_cfg[i][expert_id]
                else:
                    layer_bit_cfg[name] = args.expert_wbits
            else:
                layer_bit_cfg[name] = 16
        bit_cfg.append(layer_bit_cfg)

    return bit_cfg


class FP32GridLinear(HQQLinear):
    """Retain GPTQ grids on device moves and round only the reconstructed weight."""

    def cuda(self, device):
        scale, zero = self.meta["scale"], self.meta["zero"]
        if scale.dtype not in (torch.float16, torch.float32) or scale.dtype != zero.dtype:
            raise ValueError("GEMQ requires matching FP16 or FP32 scales and zeros")
        super().cuda(device)
        self.meta.update(scale=scale.to(device), zero=zero.to(device), compute_dtype=torch.float32)
        return self

    def dequantize(self):
        return Quantizer.dequantize(self.W_q, self.meta).to(self.compute_dtype)


def fp16_grid(value):
    stored = value.to(dtype=torch.float16)
    if not torch.isfinite(stored).all():
        raise ValueError("GPTQ grid cannot be stored as finite FP16")
    return stored


def create_hqq_linear_from_quantized_weights(
    W_q, scale, zero, org_shape, nbits, group_size, bias=None, device="cuda", compute_dtype=None, packed=False,
    fp32_dequant=False,
):
    """
    Create a HQQLinear layer from quantized weights without running on-the-fly HQQ quantization.

    NOTE: We assume the grouped quantization is performed at hidden_dim (i.e., axis=1).
    """
    # create a dummy HQQLinear layer
    # skip initialization (HQQ quantization) by passing a None linear_layer
    quant_config = BaseQuantizeConfig(nbits, group_size)
    aligned = fp32_dequant or scale.dtype == zero.dtype == torch.float32
    linear_cls = FP32GridLinear if aligned else HQQLinear
    quant_layer = linear_cls(None, quant_config, compute_dtype=compute_dtype or W_q.dtype, device=device)
    

    # store meta-data
    meta = {
        "nbits": nbits,
        "group_size": group_size,
        "shape": torch.Size(org_shape),
        "scale": fp16_grid(scale) if aligned else scale,
        "zero": fp16_grid(zero) if aligned else zero,
        "axis": 1,
        "packing": Quantizer.bit_to_packing[nbits],
        "quant_scale": False,
        "quant_zero": False,
    }
    meta["unpack_view_dtype"] = Quantizer.unpack_view_dtype[meta["packing"]]

    # pack bits
    meta["view_as_float"] = False
    if not packed:
        W_q = Quantizer.pack[meta["packing"]](W_q.float())

    # set attributes
    quant_layer.W_q = W_q
    quant_layer.meta = meta
    if bias is not None:
        quant_layer.bias = bias.data.clone().to(device=device, dtype=quant_layer.compute_dtype)

    quant_layer.cuda(device)
    quant_layer.ready = True  # to enable dequantization

    return quant_layer


def getattr_nested(obj, attr_path):
    """
    Get a nested attribute of an object given a dot-separated attribute path.
    """
    parts = attr_path.split(".")
    for p in parts:
        if p.isdigit():
            obj = obj[int(p)]  # for ModuleList
        else:
            obj = getattr(obj, p)
    return obj


def setattr_nested(obj, attr_path, value):
    """
    Set a nested attribute of an object given a dot-separated attribute path.
    """
    parts = attr_path.split(".")
    for p in parts[:-1]:
        if p.isdigit():
            obj = obj[int(p)]  # for ModuleList
        else:
            obj = getattr(obj, p)
    last = parts[-1]
    if last.isdigit():
        obj[int(last)] = value
    else:
        setattr(obj, last, value)


def replace_linears(model, model_name, quant_modules, quant_weight=True):
    """
    Replace quantized nn.Linear in model with HQQLinear modules.
    """
    layers = get_blocks(model, model_name)
    for name, m in quant_modules.items():
        # retrieve quantization params
        W = m.weight.data
        scales = m.quant_scales
        zeros = m.quant_zeros
        nbits = m.quant_nbits.item()
        group_size = m.quant_groupsize.item()

        packed = quant_weight and hasattr(m, "gemq_packed_codes")
        if packed:
            Q = m.gemq_packed_codes
        elif quant_weight:
            num_groups = scales.shape[0]
            Q = torch.round(
                W.view(num_groups, -1) / scales + zeros
            ).clamp(0, 2**nbits - 1)
        else:
            Q = W

        # Replace with HQQLinear
        device = m.weight.device if m.weight.is_cuda else "cuda"
        hqq_linear = create_hqq_linear_from_quantized_weights(
            Q, scales, zeros, m.weight.shape, nbits, group_size, bias=m.bias, device=device, compute_dtype=W.dtype, packed=packed,
            fp32_dequant=native_grid_dtype(model.config) is not None,
        )
        setattr_nested(layers, name, hqq_linear)


def check_packing(model, quant_modules, args):
    """
    Sanity check for weight packing and unpacking.
    """
    layers = get_blocks(model, args.model_name)

    max_rec_err = 0.0
    for name, m in quant_modules.items():
        qlinear = getattr_nested(layers, name)
        W_r = qlinear.dequantize()

        avg_err = (m.weight.data.cuda() - W_r.cuda()).abs().mean().item()
        max_rec_err = max(max_rec_err, avg_err)

    print(f"Max reconstruction error among all quantized linear layers: {max_rec_err}")
