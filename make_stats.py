#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/02/23 

import math
from PIL import Image
from pathlib import Path
from argparse import ArgumentParser

import numpy as np
import torch
from torch import Tensor
from torchvision.utils import make_grid
import matplotlib.pyplot as plt

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def save_avg_img(split_dp:Path, img_dp:Path, name:str) -> Tensor:
  npy_fps = list(split_dp.iterdir())

  ret = []
  uint8_max = torch.FloatTensor([255.0]).to(device)
  for cls_npy in npy_fps:    # <lbl>.npy
    X = np.load(cls_npy)                # [N, C, H, W]
    X = torch.from_numpy(X).to(device)
    if X.dtype == torch.int8:           # class-diff value is in [-128, 127]
      X = X.short()   # int16
      X = X + 128
      X = X.byte()    # uint8
    assert X.dtype == torch.uint8
    X = X.double() / uint8_max    # [0, 1]
    X = X.mean(dim=0).float()
    ret.append(X.cpu())
  X = torch.stack(ret, dim=0)

  X_grid = make_grid(X, nrow=int(len(X)**0.5), padding=2, normalize=False, scale_each=False)
  im = X_grid.permute([1, 2, 0]).numpy()
  im = (im * 255).astype(np.uint8)
  Image.fromarray(im, 'RGB').save(img_dp / f'{name}.png')

  X_grid = make_grid(X, nrow=int(len(X)**0.5), padding=2, normalize=True,  scale_each=False)
  im = X_grid.permute([1, 2, 0]).numpy()
  im = (im * 255).astype(np.uint8)
  Image.fromarray(im, 'RGB').save(img_dp / f'{name}-norm.png')

  X_grid = make_grid(X, nrow=int(len(X)**0.5), padding=2, normalize=True,  scale_each=True)
  im = X_grid.permute([1, 2, 0]).numpy()
  im = (im * 255).astype(np.uint8)
  Image.fromarray(im, 'RGB').save(img_dp / f'{name}-normeach.png')


def save_hist(split_dp:Path, img_dp:Path, name:str) -> Tensor:
  npy_fps = list(split_dp.iterdir())
  n_col = int(len(npy_fps)**0.5)
  n_row = math.ceil(len(npy_fps) / n_col)

  fig, axs = plt.subplots(n_row, n_col)
  _ = [_.axis('off') for _ in axs for _ in _]
  for idx, npy_fp in enumerate(npy_fps):
    data = np.load(npy_fp)
    ax = axs[idx // n_col][idx % n_col]
    ax.hist(data.flatten(), bins=256)
  fig.suptitle(name)
  fig.tight_layout()
  fig.savefig(img_dp / f'{name}-hist.png')


def make_stats(args):
  data_dp: Path = args.data_path / args.dataset
  img_dp:  Path = args.img_path
  img_dp.mkdir(exist_ok=True, parents=True)

  for split_dp in data_dp.iterdir():
    for type in ['single', 'cross']:
      name = f'{args.dataset}-{split_dp.name}-{type}'
      print(f'>> {name}')
      save_avg_img(split_dp / type, img_dp, name)
      save_hist(split_dp / type, img_dp, name)

  
if __name__ == '__main__':
  parser = ArgumentParser()
  parser.add_argument('-D', '--dataset', default='cifar10')
  parser.add_argument('-N', '--num_classes', default=10)
  parser.add_argument('--data_path', type=Path, default=Path('preprocessed'))
  parser.add_argument('--img_path',  type=Path, default=Path('img'))
  args = parser.parse_args()

  make_stats(args)
