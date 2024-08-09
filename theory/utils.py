import os
import sys
import time
import math

import numpy as np
from scipy.special import rel_entr
from scipy.spatial.distance import jensenshannon

import torch
import torch.nn as nn
import torch.nn.init as init


def quantize(module: nn.Module, bit, fn_filter=None):
    print(f'Bit: {bit}')
    for k, v in module.state_dict().items():
        if (fn_filter is None or fn_filter(k, v)) and v is not None:
            print(f'Quantize: {k}, #params: {torch.numel(v)}')
            HALF = int(2 ** (bit - 1) + 1e-5)
            scaled = v * HALF
            rounded = scaled.round()
            clamped = torch.clamp(rounded, -HALF, HALF-1)
            quantized = clamped / HALF
            update_dict = {k: quantized}
            module.load_state_dict(update_dict, strict=False)


def quantize_tensor(v: torch.Tensor, bit):
    print(f'Bit: {bit}, #params: {torch.numel(v)}')
    HALF = int(2 ** (bit - 1) + 1e-5)
    scaled = v * HALF
    rounded = scaled.round()
    clamped = torch.clamp(rounded, -HALF, HALF-1)
    quantized = clamped / HALF
    return quantized


def exp_linspace(start, stop, step, speed=1):
    distance = np.abs(stop - start) * speed
    stop_y = np.exp(distance)
    ys = np.linspace(1, stop_y, step)
    xs = np.log(ys) / speed
    if start < stop:
        return start + xs
    return start - xs


def exp_linspace_new(start, stop, step, speed=1):
    distance = np.abs(stop - start) * speed
    xs = np.linspace(0, distance, step)
    ys = np.exp(-xs)
    stop_y = ys[-1]
    scale_y = (start - stop) / (1. - stop_y)
    return (ys - stop_y) * scale_y + stop


def exp_linspace_newnew(start, stop, step, speed=1):
    base = np.e * speed
    # use: f(x) = (base ** -x) * start
    x_stop = np.emath.logn(base, start / stop)
    xs = np.linspace(0, x_stop, step)
    ys = (base ** -xs) * start
    return ys


def exp_linspace_newnewnew(start, stop, step, speed=1):
    # use: y = exp(-x) * speed
    x_start = -np.log(start / speed)
    x_stop = -np.log(stop / speed)
    xs = np.linspace(x_start, x_stop, step)
    ys = np.exp(-xs) * speed
    return ys


def kl_div(p, q):
    mask = np.where(q != 0)
    return np.sum(rel_entr(p[mask], q[mask]))


def js_div(p, q):
    return jensenshannon(p, q) ** 2


N_BINS_CONTINUOUS = 1000


def get_bins(bit):
    half = min(int(2 ** (bit - 1) + 1e-5), N_BINS_CONTINUOUS)
    values = np.linspace(-half, half-1, half*2)
    bins = np.array((values - 0.5).tolist() + [values[-1] + 0.5]) / half
    return bins


def get_ticks(bit):
    half = min(int(2 ** (bit - 1) + 1e-5), N_BINS_CONTINUOUS)
    values = np.linspace(-half, half-1, half*2) / half
    return values
