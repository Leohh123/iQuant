import torch
from utils import exp_linspace, quantize
import numpy as np

net = 1
args = 1

# -------- Algorithm: iQuant --------
bits = exp_linspace(32, args.bit, args.iter + 1)
for bit in bits[1:]:
    quantize(net, bit)

# -------- Function: exp_linspace --------
def exp_linspace(start, stop, step, speed=1):
    distance = np.abs(stop - start) * speed
    stop_y = np.exp(distance)
    ys = np.linspace(1, stop_y, step)
    xs = np.log(ys) / speed
    if start < stop:
        return start + xs
    return start - xs

# -------- Function: quantize --------
def quantize(module, bit):
    for k, v in module.state_dict().items():
        half = int(2 ** (bit - 1) + 1e-5)
        scaled = v * half
        rounded = scaled.round()
        clamped = torch.clamp(rounded, -half, half-1)
        quantized = clamped / half
        module.load_state_dict(
            {k: quantized}, strict=False)