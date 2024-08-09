import os
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
parser.add_argument('--check', action='store_true', default=False,
                    help='show check plot or not')
parser.add_argument('--mesh', action='store_true', default=False,
                    help='show mesh plot or not')
parser.add_argument('--save-mesh', action='store_true', default=False,
                    help='save mesh plot or not')
parser.add_argument('--arch', type=str, required=True, metavar='A',
                    help='architecture of the model')
parser.add_argument('--weight', type=str, required=True, metavar='W',
                    help='weights of the model')
parser.add_argument('--sbit', type=float, default=10, metavar='SB',
                    help='quantization bit (source)')
parser.add_argument('--tbit', type=float, default=1, metavar='TB',
                    help='quantization bit (target)')
parser.add_argument('--speed', type=float, default=0.15, metavar='S',
                    help='speed of bits')
parser.add_argument('--nbit', type=int, default=10, metavar='NB',
                    help='#quantization bits')
parser.add_argument('--niter', type=int, default=10, metavar='NI',
                    help='#(quantization #iterations)')
args = parser.parse_args()


if args.task == 'bit-iter-acc1-acc5-kl-js-th-qr':
    bits = exp_linspace(args.sbit, args.tbit, args.nbit, speed=args.speed)
    result_name = f'bit-iter-acc1-acc5-kl-js-th-qr.csv'
    if args.run:
        if args.check:
            plt.plot(bits, 'o-')
            plt.show()
        for bit in bits:
            for it in range(1, args.niter + 1):
                subprocess.run([
                    'python', 'quant.py',
                    '-net', args.arch,
                    '-weights', args.weight,
                    '--bit', str(bit),
                    '--iter', str(it),
                    '--result-name', result_name,
                    '--result-cmd', "f'{args.bit},{args.iter},{top1_acc},{top5_acc},{KLs[-1]},{JSs[-1]},{quant_thresh},{quant_rate}\\n'",
                    '--save-fig',
                    '--save-param',
                ])

    if args.mesh:
        data = np.genfromtxt(os.path.join(
            f'./results/{args.arch}/', result_name), delimiter=',')

        X = bits
        Y = np.arange(1, args.niter + 1)
        X, Y = np.meshgrid(X, Y)
        Z_acc1 = np.zeros_like(X)
        Z_acc5 = np.zeros_like(X)
        Z_kl = np.zeros_like(X)
        Z_js = np.zeros_like(X)
        # print(X)

        def find_nearest(array, value):
            array = np.asarray(array)
            idx = (np.abs(array - value)).argmin()
            return idx

        # cnt = 0
        for k in range(data.shape[0]):
            x, y = find_nearest(bits, data[k, 0]), int(data[k, 1]) - 1
            # Z_acc1[y, x] = cnt
            # cnt += 1
            Z_acc1[y, x] = data[k, 2]
            Z_acc5[y, x] = data[k, 3]
            Z_kl[y, x] = data[k, 4]
            Z_js[y, x] = data[k, 5]
            # print(y, x, data[k, 2], data[k, 3], data[k, 4])

        fig = plt.figure(figsize=plt.figaspect(0.25))

        ax1 = fig.add_subplot(1, 4, 1, projection='3d')
        surf1 = ax1.plot_surface(
            X, Y, Z_acc1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
        ax1.set_xlabel('Bits')
        ax1.set_ylabel('# Iterations')
        ax1.set_zlabel('Acc (top 1)')
        ax1.azim = -170
        fig.colorbar(surf1, shrink=0.5, aspect=5)

        ax2 = fig.add_subplot(1, 4, 2, projection='3d')
        surf2 = ax2.plot_surface(
            X, Y, Z_acc5, cmap=cm.coolwarm, linewidth=0, antialiased=False)
        ax2.set_xlabel('Bits')
        ax2.set_ylabel('# Iterations')
        ax2.set_zlabel('Acc (top 5)')
        ax2.azim = -170
        fig.colorbar(surf2, shrink=0.5, aspect=5)

        ax3 = fig.add_subplot(1, 4, 3, projection='3d')
        surf3 = ax3.plot_surface(
            X, Y, Z_kl, cmap=cm.coolwarm, linewidth=0, antialiased=False)
        ax3.set_xlabel('Bits')
        ax3.set_ylabel('# Iterations')
        ax3.set_zlabel('KL divergence')
        ax3.azim = -170
        fig.colorbar(surf3, shrink=0.5, aspect=5)

        ax4 = fig.add_subplot(1, 4, 4, projection='3d')
        surf4 = ax4.plot_surface(
            X, Y, Z_js, cmap=cm.coolwarm, linewidth=0, antialiased=False)
        ax4.set_xlabel('Bits')
        ax4.set_ylabel('# Iterations')
        ax4.set_zlabel('JS divergence')
        ax4.azim = -170
        fig.colorbar(surf4, shrink=0.5, aspect=5)

        fig.suptitle(args.arch)

        if args.save_mesh:
            os.makedirs('./figures/all/', exist_ok=True)
            plt.savefig(f'./figures/all/{args.arch}.png', dpi=500)
        else:
            plt.show()

# python plotone.py --task bit-iter-acc1-acc5-kl-js-th-qr --run --check --mesh --arch mobilenet --weight ./checkpoint/mobilenet/Friday_02_August_2024_03h_54m_59s/mobilenet-162-best.pth
