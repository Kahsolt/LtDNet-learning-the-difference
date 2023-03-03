#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/02/23 

from pathlib import Path
import random
import gc

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np


class ClassDiffDataset(Dataset):

  def __init__(self, root:Path, split:str='train'):
    super().__init__()

    self.base_path = root / split / 'cross'

    X, Y = [], []
    for npy_fn in self.base_path.iterdir():
      data = np.load(npy_fn)
      label = np.ones([len(data)], dtype=np.uint8) * int(npy_fn.stem)
      X.append(data)
      Y.append(label)

    self.X = np.concatenate(X, axis=0)  # int8,  [-128, 127]
    self.Y = np.concatenate(Y, axis=0)  # uint8, [0, NC**2-1]
    del X, Y ; gc.collect()

    print('X.shape:', self.X.shape)
    print('X.dtype:', self.X.dtype)
    print('Y.shape:', self.Y.shape)
    print('Y.dtype:', self.Y.dtype)

  def __len__(self):
    return len(self.Y)

  def __getitem__(self, idx):
    x = self.X[idx].astype(np.float32)    # [-128.0, 127.0]
    x = (x + 0.5) / 127.5                 # float32, [-1.0, 1.0]
    y = self.Y[idx].astype(np.int64)
    return x, y


class ClassSampleDataset:

  def __init__(self, root:Path, split:str='train'):
    super().__init__()

    self.base_path = root / split / 'single'

    self.samples = {
      int(npy_fn.stem): np.load(npy_fn)
        for npy_fn in self.base_path.iterdir()
    }

  @property
  def num_classes(self):
    return len(self.samples)
  
  def get_samples_of_class(self, label:int, count=10):
    nlen = len(self.samples[label])
    idx = random.sample(range(nlen), k=count)
    return self.samples[label][idx, ...]     # int8


if __name__ == '__main__':
  dataset = ClassDiffDataset(Path('preprocessed') / 'cifar10', split='valid')
  dataloader = DataLoader(dataset, batch_size=192, shuffle=True)

  for X, Y in dataloader:
    print(X.shape)
    print(Y.shape)

    breakpoint()
