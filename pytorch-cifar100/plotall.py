import os
import subprocess
import argparse


parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--plot', action='store_true', default=False,
                    help='plot figures of all models')
parser.add_argument('--save', action='store_true', default=False,
                    help='save figures of all models')
args = parser.parse_args()


PATH_CHECKPOINT = './checkpoint/'
PATH_PLOT = './figures/all/'

all_archs = os.listdir(PATH_CHECKPOINT)
ploted_archs = list(map(lambda s: s.rstrip('.png'), os.listdir(PATH_PLOT)))
ploting_archs = sorted(list(set(all_archs).difference(ploted_archs)))
# print(f'{all_archs=}')
# print(f'{ploted_archs=}')
# print(f'{ploting_archs=}')
# exit()


for arch in ploting_archs:
    path_arch = os.path.join(PATH_CHECKPOINT, arch)
    # print(1, arch, path_arch)
    # continue
    for runtime in os.listdir(path_arch):
        path_runtime = os.path.join(path_arch, runtime)
        best_models = sorted(filter(
            lambda s: s.endswith('best.pth'), os.listdir(path_runtime)
        ), key=lambda s: int(s.split('-')[1]), reverse=True)
        # print(1, runtime, path_runtime, best_models)
        path_best = os.path.join(path_runtime, best_models[0])
        print(arch, path_best)
        run_args = [
            'python', 'plotone.py',
            '--task', 'bit-iter-acc1-acc5-kl-js-th-qr',
            '--arch', arch,
            '--weight', path_best,
            '--sbit', 10,
            '--tbit', 1,
            '--speed', 0,
            '--nbit', 19,
            '--niter', 20,
            # '--check',
        ]
        run_args = list(map(str, run_args))
        if args.save:
            subprocess.run(run_args + [
                '--mesh',
                '--save-mesh',
            ])
        elif args.plot:
            subprocess.run(run_args + [
                '--check',
                '--mesh',
            ])
        else:
            subprocess.run(run_args + [
                '--run',
            ])
