import os
import time
import argparse

from matplotlib import pyplot as plt

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import numpy as np

from conf import settings
from utils import get_network, get_test_dataloader, quantize, exp_linspace, get_bins, N_BINS_CONTINUOUS, kl_div, js_div, quant_thresh_and_rate

if __name__ == '__main__':

    # ================ Load ================

    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, required=True, help='net type')
    parser.add_argument('-weights', type=str, required=True,
                        help='the weights file you want to test')
    parser.add_argument('-gpu', action='store_true',
                        default=True, help='use gpu or not')
    parser.add_argument('-b', type=int, default=16,
                        help='batch size for dataloader')
    parser.add_argument('--bit', type=float, required=False, metavar='B',
                        help='quantization bit (int or float)')
    parser.add_argument('--iter', type=int, default=0, metavar='I',
                        help='quantization #iterations')
    parser.add_argument('--show-fig', action='store_true', default=False,
                        help='show figures')
    parser.add_argument('--save-fig', action='store_true', default=False,
                        help='save figures')
    parser.add_argument('--save-param', action='store_true', default=False,
                        help='save parameters over time')
    parser.add_argument('--result-name', type=str, default=None, metavar='RN',
                        help='filename of saving results')
    parser.add_argument('--result-cmd', type=str, default=None, metavar='RC',
                        help='command string of generating results')
    args = parser.parse_args()

    net = get_network(args)

    cifar100_test_loader = get_test_dataloader(
        settings.CIFAR100_TRAIN_MEAN,
        settings.CIFAR100_TRAIN_STD,
        # settings.CIFAR100_PATH,
        num_workers=4,
        batch_size=args.b,
    )

    net.load_state_dict(torch.load(args.weights))
    # print(net)

    # ================ iQuant ================

    # Make directories
    dirnames = ['figures', 'results', 'params']
    for dirname in dirnames:
        path = os.path.join(dirname, args.net)
        os.makedirs(path, exist_ok=True)
        print(path)

    quant_thresh, quant_rate = quant_thresh_and_rate(net)
    print(f'Quantization threshold: {quant_thresh}')
    N_SAMPLE = 100

    indexes = {}

    def get_index(v):
        numel = torch.numel(v)
        if numel in indexes:
            return indexes[numel]
        index = torch.randint(0, numel, (N_SAMPLE,))
        indexes[numel] = index
        return index

    def fn_filter(k, v):
        return torch.numel(v) >= quant_thresh

    hist_params = []

    def get_params():
        params = []
        for k, v in net.state_dict().items():
            if fn_filter(k, v):
                flattened = v.detach().cpu().flatten()
                index = get_index(flattened)
                params.extend(flattened[index].tolist())
                # print(f'{k=}, {v=}')
        return params

    if args.bit is not None:
        bits = exp_linspace(32, args.bit, args.iter + 1)
        print(bits)
        for bit in bits[1:]:
            hist_params.append(get_params())
            quantize(net, bit, fn_filter)
        hist_params.append(get_params())
    else:
        print('No quantization')

    # ================ Test ================

    correct_1 = 0.0
    correct_5 = 0.0
    total = 0

    net.eval()
    with torch.no_grad():
        for n_iter, (image, label) in enumerate(cifar100_test_loader):
            # print("iteration: {}\ttotal {} iterations".format(
            #     n_iter + 1, len(cifar100_test_loader)))

            if args.gpu:
                image = image.cuda()
                label = label.cuda()
                # print('GPU INFO.....')
                # print(torch.cuda.memory_summary(), end='')

            output = net(image)
            _, pred = output.topk(5, 1, largest=True, sorted=True)

            label = label.view(label.size(0), -1).expand_as(pred)
            correct = pred.eq(label).float()

            # compute top 5
            correct_5 += correct[:, :5].sum()

            # compute top1
            correct_1 += correct[:, :1].sum()

    top1_acc = (correct_1 * 100. / len(cifar100_test_loader.dataset)).item()
    top5_acc = (correct_5 * 100. / len(cifar100_test_loader.dataset)).item()
    num_params = sum(p.numel() for p in net.parameters())

    print("Top 1 err: ", top1_acc)
    print("Top 5 err: ", top5_acc)
    print("Parameter numbers: {}".format(num_params))

    # ================ Plot ================

    if args.save_fig or args.show_fig:
        # plot distribution over time
        Y = np.array(hist_params).T
        x = np.arange(1, len(hist_params) + 1, 1)
        # print(f'{x.shape=}')
        # print(f'{x=}')
        # print(f'{Y.shape=}')
        # print(f'{Y=}')

        num_fine = 800
        x_fine = np.linspace(x.min(), x.max(), num_fine)
        y_fine = np.concatenate([np.interp(x_fine, x, y_row) for y_row in Y])
        x_fine = np.broadcast_to(x_fine, (Y.shape[0], num_fine)).ravel()

        fig, axes = plt.subplots(
            nrows=3, ncols=2, figsize=(12, 10), layout='constrained')
        # 1)
        axes[0, 0].plot(x, Y.T, color='C0', alpha=0.1)
        axes[0, 0].set_ylim(-0.2, 0.2)
        axes[0, 0].set_xmargin(0)
        axes[0, 0].set_title('Line plot with alpha')
        # 2)
        cmap = plt.colormaps['plasma']
        cmap = cmap.with_extremes(bad=cmap(0))
        h, xedges, yedges = np.histogram2d(x_fine, y_fine, bins=[400, 200])
        pcm = axes[1, 0].pcolormesh(xedges, yedges, h.T, cmap=cmap, norm='log',
                                    rasterized=True)
        fig.colorbar(pcm, ax=axes[1, 0], label='# points', pad=0)
        axes[1, 0].set_facecolor(cmap(0))
        axes[1, 0].set_ylim(-0.2, 0.2)
        axes[1, 0].set_title('2d histogram and log color scale')
        # 3)
        pcm = axes[2, 0].pcolormesh(xedges, yedges, h.T, cmap=cmap,
                                    rasterized=True)
        fig.colorbar(pcm, ax=axes[2, 0], label='# points', pad=0)
        axes[2, 0].set_facecolor(cmap(0))
        axes[2, 0].set_ylim(-0.2, 0.2)
        axes[2, 0].set_title('2d histogram and linear color scale')
        # 4)

        axes[0, 1].hist(hist_params[0], bins=N_BINS_CONTINUOUS, density=True,
                        histtype='bar', alpha=.5, label='initial')
        bins = get_bins(args.bit)
        axes[0, 1].hist(hist_params[-1], bins=bins, density=True,
                        histtype='bar', alpha=.5, label='quantized')
        axes[0, 1].set_xlim(-0.2, 0.2)
        axes[0, 1].legend()
        axes[0, 1].set_yscale('log')
        axes[0, 1].set_title('Parameters probability density')
        # 5)
        distribs = []
        for i, bit in enumerate(bits):
            bins = get_bins(bit)
            p = np.histogram(hist_params[0], bins=bins, density=True)[0]
            q = np.histogram(hist_params[i], bins=bins, density=True)[0]
            # print(i, bit, p.shape, q.shape)
            distribs.append([p, q])
        # print('distribs', distribs)
        KLs = [kl_div(p, q) for p, q in distribs]
        JSs = [js_div(p, q) for p, q in distribs]
        # print('JSs', JSs)
        axes[1, 1].plot(x, KLs, 'o-')
        axes[1, 1].set_xmargin(0)
        axes[1, 1].set_title('KL divergence over iterations')
        # 5)
        axes[2, 1].plot(x, JSs, 'o-')
        axes[2, 1].set_xmargin(0)
        axes[2, 1].set_title('JS divergence over iterations')

        # color = 'tab:red'
        # axes[4].plot(x, KLs, 'o-', color=color)
        # axes[4].set_xmargin(0)
        # axes[4].set_ylabel('KL divergences', color=color)
        # axes[4].tick_params(axis='y', labelcolor=color)
        # axes[4].set_title('Divergences over iterations')
        # color = 'tab:blue'
        # ax4 = axes[4].twinx()
        # ax4.plot(x, JSs, 'o-', color=color)
        # ax4.set_ylabel('JS divergences', color=color)
        # ax4.tick_params(axis='y', labelcolor=color)

        result_id = f'B={args.bit:.2f},I={args.iter}'

        if args.save_fig:
            plt.savefig(
                f'./figures/{args.net}/{result_id}.png', dpi=500)
        elif args.show_fig:
            plt.show()

        if args.save_param:
            with open(f'./params/{args.net}/{result_id}.npy', 'wb') as f:
                np.save(f, np.array(hist_params))

        if args.result_name is not None:
            with open(os.path.join(f'./results/{args.net}/', args.result_name), 'a+') as f:
                f.write(eval(args.result_cmd))

# python quant.py -net mobilenet -weights ./checkpoint/mobilenet/Friday_02_August_2024_03h_54m_59s/mobilenet-162-best.pth --bit 4
