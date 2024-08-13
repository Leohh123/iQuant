import os
import sys
import subprocess
from datetime import datetime
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--source-dir', '-s', type=str, required=True,
                    help='source directory')
parser.add_argument('--target-dir', '-t', type=str, required=True,
                    help='target directory')
args = parser.parse_args()

assert os.path.isdir(args.source_dir) and os.path.isdir(args.target_dir), \
    'Invalid directories'
assert all(map(lambda d: os.path.isdir(os.path.join(args.source_dir, d)), os.listdir(args.source_dir))), \
    'Unexpected files in the source directory'
assert all(map(lambda d: os.path.isfile(os.path.join(args.target_dir, d)), os.listdir(args.target_dir))), \
    'Unexpected directories in the target directory'

source_name = os.path.basename(os.path.abspath(args.source_dir))
time_now = datetime.now()
time_str = time_now.strftime('%m%d%H%M')
archive_name = f'{source_name}_{time_str}.tar'
archive_path = os.path.join(args.target_dir, archive_name)
assert not os.path.exists(archive_path), 'Archive already exists'

filtered_paths = []
print('Sources:')
for d in os.listdir(args.source_dir):
    path = os.path.join(args.source_dir, d)
    mtime = os.path.getmtime(path)
    delta = time_now.timestamp() - mtime
    if delta > 60 * 60:
        filtered_paths.append(path)
        print(datetime.fromtimestamp(mtime).strftime('[%Y/%m/%d %H:%M]'), path)

print(f'Target: {archive_path}')

if input('Continue? y/[n]') == 'y':
    subprocess.run(['tar', '-cvf', archive_path, *filtered_paths])
    print('Done.')
else:
    print('Stopped.')
