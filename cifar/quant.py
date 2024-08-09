'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import numpy as np
from matplotlib import pyplot as plt

import os
import argparse

from models import *
from utils import progress_bar, quantize, exp_linspace, kl_div, js_div, get_bins, N_BINS_CONTINUOUS


torch.manual_seed(1)


# Make directories
dirnames = ['checkpoint', 'figures', 'results', 'params']
for dirname in dirnames:
    if not os.path.isdir(dirname):
        os.mkdir(dirname)


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--load-model', type=str, required=False, metavar='L',
                    help='load model from checkpoint')
parser.add_argument('--bit', type=float, required=False, metavar='B',
                    help='quantization bit (int or float)')
parser.add_argument('--iter', type=int, default=0, metavar='I',
                    help='quantization #iterations')
parser.add_argument('--save-fig', action='store_true', default=False,
                    help='save figures')
parser.add_argument('--save-param', action='store_true', default=False,
                    help='save parameters over time')
parser.add_argument('--result-name', type=str, default=None, metavar='RN',
                    help='filename of saving results')
parser.add_argument('--result-cmd', type=str, default=None, metavar='RC',
                    help='command string of generating results')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=0)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=0)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')
net = VGG('VGG19')
# net = ResNet18()
# net = PreActResNet18()
# net = GoogLeNet()
# net = DenseNet121()
# net = ResNeXt29_2x64d()
# net = MobileNet()
# net = MobileNetV2()
# net = DPN92()
# net = ShuffleNetG2()
# net = SENet18()
# net = ShuffleNetV2(1)
# net = EfficientNetB0()
# net = RegNetX_200MF()
# net = SimpleDLA()
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

if args.load_model:
    # Load checkpoint.
    print('==> Loading model..')
    checkpoint = torch.load(args.load_model)
    net.load_state_dict(checkpoint['net'])

criterion = nn.CrossEntropyLoss()


def test():
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    # if acc > best_acc:
    #     print('Saving..')
    #     state = {
    #         'net': net.state_dict(),
    #         'acc': acc,
    #         'epoch': epoch,
    #     }
    #     if not os.path.isdir('checkpoint'):
    #         os.mkdir('checkpoint')
    #     torch.save(state, './checkpoint/ckpt.pth')
    #     best_acc = acc

    print(f'Accuracy: {acc:.2f} ({correct}/{total})')
    return acc


QUANT_THRESH = 100000
N_SAMPLE = 1000

indexes = {}


def get_index(v):
    numel = torch.numel(v)
    if numel in indexes:
        return indexes[numel]
    index = torch.randint(0, numel, (N_SAMPLE,))
    indexes[numel] = index
    return index


def fn_filter(k, v):
    return 'features' in k and 'weight' in k and torch.numel(v) >= QUANT_THRESH


hist_params = []


def get_params():
    params = []
    for k, v in net.state_dict().items():
        if fn_filter(k, v):
            flattened = v.detach().cpu().flatten()
            index = get_index(flattened)
            params.extend(flattened[index].tolist())
    return params


if args.bit is not None:
    bits = exp_linspace(32, args.bit, args.iter + 1)
    for bit in bits[1:]:
        hist_params.append(get_params())
        quantize(net, bit, fn_filter)
    hist_params.append(get_params())
else:
    print('No quantization')

# for epoch in range(start_epoch, start_epoch+200):
#     train(epoch)
#     test(epoch)
#     scheduler.step()

accuracy = test()


# plot distribution over time
Y = np.array(hist_params).T
x = np.arange(1, len(hist_params) + 1, 1)

num_fine = 800
x_fine = np.linspace(x.min(), x.max(), num_fine)
y_fine = np.concatenate([np.interp(x_fine, x, y_row) for y_row in Y])
x_fine = np.broadcast_to(x_fine, (Y.shape[0], num_fine)).ravel()

fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(12, 10), layout='constrained')
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


result_id = f'B={args.bit},I={args.iter}'


if args.save_fig:
    plt.savefig(
        f'./figures/{result_id}.png', dpi=500)
else:
    plt.show()

if args.save_param:
    with open(f'./params/{result_id}.npy', 'wb') as f:
        np.save(f, np.array(hist_params))

if args.result_name is not None:
    with open(args.result_name, 'a+') as f:
        f.write(eval(args.result_cmd))
