# import numpy as np
# from scipy.special import rel_entr, kl_div


# def KL(a, b):
#     a = np.asarray(a, dtype=float)
#     b = np.asarray(b, dtype=float)
#     return np.sum(np.where(a != 0, a * np.log(a / b), 0))


# p = np.array([1.346112, 1.337432, 1.246655, 1]) * 2
# q = np.array([1.033836, 1.082015, 1.117323, 0]) * 3

# print(KL(p, q))
# print(rel_entr(p, q).sum())
# print(kl_div(p, q).sum())


import matplotlib.pyplot as plt
import numpy as np

from matplotlib import cm
from mpl_toolkits.mplot3d.axes3d import get_test_data

# set up a figure twice as wide as it is tall
fig = plt.figure(figsize=plt.figaspect(0.5))

# =============
# First subplot
# =============
# set up the Axes for the first plot
ax = fig.add_subplot(1, 2, 1, projection='3d')

# plot a 3D surface like in the example mplot3d/surface3d_demo
x = np.arange(-5, 5, 0.5)
y = np.arange(-5, 5, 0.25)
X, Y = np.meshgrid(x, y)
Z = np.zeros_like(X)
cnt = 0
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        Z[i, j] = cnt
        cnt += 1
print(Z)
surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
ax.set_xlabel('x')
ax.set_ylabel('y')
fig.colorbar(surf, shrink=0.5, aspect=10)

# ==============
# Second subplot
# ==============
# set up the Axes for the second plot
ax = fig.add_subplot(1, 2, 2, projection='3d')

# plot a 3D wireframe like in the example mplot3d/wire3d_demo
X, Y, Z = get_test_data(0.05)
ax.plot_wireframe(X, Y, Z, rstride=10, cstride=10)

plt.show()


# import matplotlib.pyplot as plt
# import numpy as np


# fig, axes = plt.subplots(nrows=6, figsize=(6, 8), layout='constrained')


# # Create some mock data
# t = np.arange(0.01, 10.0, 0.01)
# data1 = np.exp(t)
# data2 = np.sin(2 * np.pi * t)

# color = 'tab:red'
# axes[0].set_xlabel('time (s)')
# axes[0].set_ylabel('exp', color=color)
# axes[0].plot(t, data1, color=color)
# axes[0].tick_params(axis='y', labelcolor=color)

# ax2 = axes[0].twinx()  # instantiate a second Axes that shares the same x-axis

# color = 'tab:blue'
# ax2.set_ylabel('sin', color=color)  # we already handled the x-label with ax1
# ax2.plot(t, data2, color=color)
# ax2.tick_params(axis='y', labelcolor=color)

# # fig.tight_layout()  # otherwise the right y-label is slightly clipped
# plt.show()
