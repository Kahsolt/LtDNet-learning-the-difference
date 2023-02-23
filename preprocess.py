#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/02/23 

from pathlib import Path
from argparse import ArgumentParser
from collections import defaultdict

import random ; random.seed(114514)
import numpy as np
import torchvision.datasets as D


def get_d(x: np.ndarray, y: np.ndarray, dtype=np.int8) -> np.ndarray:
  x = x.astype(np.int16)    # int16
  y = y.astype(np.int16)
  d = x - y                 # [-511, 511]
  assert -512 < d.min() and d.max() < 512
  d //= 4                   # rescale `d` for saving as int8 to save disk space...
  assert np.iinfo(dtype).min < d.min() and d.max() < np.iinfo(dtype).max
  d = d.astype(np.int8)     # int8
  return d

def save_shard(data:np.ndarray, fp:Path):
  fp.parent.mkdir(exist_ok=True, parents=True)
  np.save(fp, data)


def preprocess_cifar10(args, num_classes=10):
  ''' airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck '''

  splits = {
    'valid': lambda: D.CIFAR10(root=args.rdata_path, train=False, download=True),
    'train': lambda: D.CIFAR10(root=args.rdata_path, train=True,  download=True),
  }

  for split, dataset in splits.items():
    log(f'[{split}]')

    log('>> original images:')
    samples = defaultdict(list)   # {int: [ndarray]}
    for x, y in dataset():
      lbl = y
      img = np.asarray(x, dtype=np.uint8).transpose([2, 0, 1])   # [C, H, W]
      samples[lbl].append(img)

    n_total = sum([len(sub) for sub in samples.values()])
    n_avg = n_total // num_classes
    log(f'total samples: {n_total}')
    log(f'average samples for each class: {n_avg}')
    for i in sorted(samples.keys()):
      log(f'  class-{i}: {len(samples[i])} ({len(samples[i]) / n_total:.3%})')
    
    for lbl, imgs in samples.items():
      save_shard(np.stack(imgs, axis=0), args.out_dp / split / 'single' / f'{lbl}.npy')

    lblmax = (num_classes - 1) ** 2
    lbllen = len(str(lblmax))
    log('>> cross-class difference images:')
    for i, imgi in samples.items():
      for j, imgj in samples.items():
        idx_grid = [ (x, y) for x in range(len(imgi)) for y in range(len(imgj)) ]
        idx_sel = random.sample(idx_grid, int(args.ratio * n_avg))
        imgx = [ get_d(imgi[x], imgj[y]) for x, y in idx_sel ]
        lblx = i * num_classes + j     # NC-based number
        lblx: str = str(lblx).rjust(lbllen, '0')

        log(f'  class-{lblx} ({i} x {j}): {len(imgx)}')
        save_shard(np.stack(imgx, axis=0), args.out_dp / split / 'cross' / f'{lblx}.npy')


if __name__ == '__main__':
  PREPROCESSORS = [x.split('_')[-1] for x in globals() if x.startswith('preprocess_')]

  parser = ArgumentParser()
  parser.add_argument('-D', '--dataset', default='cifar10', choices=PREPROCESSORS)
  parser.add_argument('-R', '--ratio', type=float, default=10, help='resample ratio for each crossed-class')    # := 交叉类的样本数 / 原始类平均样本数
  parser.add_argument('--rdata_path',  type=Path,  default=Path('data'))
  parser.add_argument('--data_path',   type=Path,  default=Path('preprocessed'))
  args = parser.parse_args()

  proc = globals().get(f'preprocess_{args.dataset}')
  assert proc

  args.rdata_path.mkdir(exist_ok=True)
  args.data_path. mkdir(exist_ok=True)
  with open(args.data_path / f'{args.dataset}-stats.txt', 'w', encoding='utf-8') as fh:
    def log(s:str=''):
      fh.write(s) ; fh.write('\n')
      print(s)
    
    args.out_dp = args.data_path / args.dataset
    proc(args)
