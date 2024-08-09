import argparse
import subprocess
import numpy as np
from matplotlib import cm
from matplotlib import pyplot as plt
from utils import exp_linspace

parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--task', type=str, required=True, metavar='T',
                    help='which plot task')
parser.add_argument('--run', action='store_true', default=False,
                    help='run quantization program or not')
args = parser.parse_args()

if args.task == 'bit-iter-acc-kl-js':
    N_ITER = 10
    bits = exp_linspace(10, 1, 10, speed=0.15)
    if args.run:
        plt.plot(bits, 'o-')
        plt.show()
        for bit in bits:
            for it in range(1, N_ITER + 1):
                subprocess.run([
                    'python', 'quant.py',
                    '--load-model', './checkpoint/VGG-VGG19.pt',
                    '--result-name', './results/bit-iter-acc-kl-js.csv',
                    '--result-cmd', "f'{args.bit},{args.iter},{accuracy},{KLs[-1]},{JSs[-1]}\\n'",
                    '--save-fig',
                    '--save-param',
                    '--bit', str(bit),
                    '--iter', str(it)
                ])
    data = np.genfromtxt('./results/bit-iter-acc-kl-js.csv', delimiter=',')

    X = bits
    Y = np.arange(1, N_ITER + 1)
    X, Y = np.meshgrid(X, Y)
    Z_acc = np.zeros_like(X)
    Z_kl = np.zeros_like(X)
    Z_js = np.zeros_like(X)
    print(X)

    def find_nearest(array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return idx

    # cnt = 0
    for k in range(data.shape[0]):
        x, y = find_nearest(bits, data[k, 0]), int(data[k, 1]) - 1
        # Z_acc[y, x] = cnt
        # cnt += 1
        Z_acc[y, x] = data[k, 2]
        Z_kl[y, x] = data[k, 3]
        Z_js[y, x] = data[k, 4]
        print(y, x, data[k, 2], data[k, 3], data[k, 4])

    fig = plt.figure(figsize=plt.figaspect(0.33))

    ax1 = fig.add_subplot(1, 3, 1, projection='3d')
    surf1 = ax1.plot_surface(
        X, Y, Z_acc, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    ax1.set_xlabel('Bits')
    ax1.set_ylabel('# Iterations')
    ax1.set_zlabel('Accuracy')
    fig.colorbar(surf1, shrink=0.5, aspect=5)

    ax2 = fig.add_subplot(1, 3, 2, projection='3d')
    surf2 = ax2.plot_surface(
        X, Y, Z_kl, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    ax2.set_xlabel('Bits')
    ax2.set_ylabel('# Iterations')
    ax2.set_zlabel('KL divergence')
    fig.colorbar(surf2, shrink=0.5, aspect=5)

    ax3 = fig.add_subplot(1, 3, 3, projection='3d')
    surf3 = ax3.plot_surface(
        X, Y, Z_js, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    ax3.set_xlabel('Bits')
    ax3.set_ylabel('# Iterations')
    ax3.set_zlabel('JS divergence')
    fig.colorbar(surf3, shrink=0.5, aspect=5)

    plt.show()
