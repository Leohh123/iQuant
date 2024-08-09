from matplotlib import pyplot as plt
import numpy as np
from utils import exp_linspace_newnewnew as exp_linspace


for i in range(1, 10):
    y = exp_linspace(32, 8, step=10, speed=i/5)
    plt.plot(y, '-o')
plt.show()

for step in range(2, 10):
    y = exp_linspace(32, 8, step=step, speed=0.5)
    plt.plot(np.linspace(0, 1, step), y, '-o')
plt.show()

y1 = exp_linspace(32, 8, step=10, speed=0.5)
y2 = exp_linspace(32, 4, step=10, speed=0.5)
plt.plot(y1)
plt.plot(y2)
plt.show()

bits = exp_linspace(32, 8, step=10, speed=1)
halfs = bits / 2
ds = halfs[1:] - halfs[:-1]
plt.plot(ds)
plt.show()
