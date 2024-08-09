import argparse

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from matplotlib import pyplot as plt
from utils import quantize, quantize_tensor, exp_linspace, get_ticks, get_bins


parser = argparse.ArgumentParser(description='Mapping test')
parser.add_argument('-b1', type=float)
parser.add_argument('-b2', type=float)
parser.add_argument('--iter', '-i', type=int, default=0)
args = parser.parse_args()


bits = exp_linspace(args.b1, args.b2, args.iter + 1)
# bits = np.arange(args.b1, args.b2-1, -1)

# plt.plot(bits)
# plt.show()

v = torch.tensor(get_ticks(args.b1))
t = quantize_tensor(v, args.b2)
qs = [v]
dqs = [torch.zeros_like(v)]
for bit in bits[1:]:
    q = quantize_tensor(qs[-1], bit)
    dq = (v - q)
    qs.append(q)
    dqs.append(dq)

fig, axes = plt.subplots(ncols=(len(bits)-1), figsize=(
    4*(len(bits)-1), 4), layout='constrained')
# for i in range(1, len(bits)):
    # axes[i - 1].plot(v, qs[i] - t, label=f'q{i} - t')
    # axes[i - 1].legend()
    # axes[i - 1].set_xlim(np.array([-2**-bits[i], 2**-bits[i]]) * 5)

    # axes[i].plot(dqs[i], '-o', label='Delta Q')

for i in range(1,len(bits)):
    axes[i-1].hist(qs[i], bins=get_bins(bits[i]))
    axes[i-1].set_xlim(np.array([-2**(1-bits[i]), 2**(1-bits[i])]) * 5)
    axes[i-1].set_title(f'{bits[i-1]:.2f} => {bits[i]:.2f}')
plt.show()
print(get_ticks(args.b2))
