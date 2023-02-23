#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/02/23 

from pathlib import Path

from torch.utils.data import Dataset, DataLoader
import numpy as np


class ClassDiffDataset(Dataset):

  def __init__(self, root:Path, split:str='train'):
    super().__init__()

    self.base_path = root / split / 'cross'

    X, Y = [], []
    for npy_fn in self.base_path.iterdir():
      data = np.load(npy_fn)
      label = np.ones([len(data)], dtype=np.int16) * int(npy_fn.stem)
      X.append(data)
      Y.append(label)

    self.X = np.concatenate(X, axis=0)
    self.Y = np.concatenate(Y, axis=0)
    del X, Y

    print('X.shape:', self.X.shape)
    print('Y.shape:', self.Y.shape)

  def __len__(self):
    return len(self.Y)

  def __getitem__(self, idx):
    return self.X[idx], self.Y[idx]


if __name__ == '__main__':
  dataset = ClassDiffDataset(Path('preprocessed') / 'cifar10', split='valid')
  dataloader = DataLoader(dataset, batch_size=192, shuffle=True)

  for X, Y in dataloader:
    print(X.shape)
    print(Y.shape)

    breakpoint()
