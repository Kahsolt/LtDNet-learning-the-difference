#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/02/23 

from argparse import ArgumentParser

import random
import numpy as np
import torch
from torchvision.utils import make_grid
import matplotlib.pyplot as plt


def show_npy(args):
  data = np.load(args.file)

  if args.random:
    data = random.sample(data, args.count)
    data = np.stack(data, axis=0)
  else:
    data = data[:args.count, ...]

  data = data / 255.0
  X = torch.from_numpy(data)
  X_grid = make_grid(X, nrow=int(args.count**0.5), padding=2, normalize=True, scale_each=True)
  im = X_grid.permute([1, 2, 0]).numpy()
  plt.imshow(im)
  plt.show()


if __name__ == '__main__':
  parser = ArgumentParser()
  parser.add_argument('-F', '--file', required=True, help='path to preprocessed *.npy file')
  parser.add_argument('-N', '--count', default=100, type=int)
  parser.add_argument('-R', '--random', action='store_true')
  args = parser.parse_args()

  show_npy(args)
