#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/03/03

from time import time
from pathlib import Path
from argparse import ArgumentParser

import torch
import numpy as np
from tqdm import tqdm

from data import ClassDiffDataset, DataLoader
from model import ResNet18_32x32

from test import get_pred

device = 'cuda' if torch.cuda.is_available() else 'cpu'


@torch.inference_mode()
def test(args):
  print('>> [Test]')

  ''' Model '''
  model = ResNet18_32x32(num_classes=100, pretrained=False).to(device)

  fp: Path = args.log_path / args.exp_name / 'model-best.pth'
  state_dict = torch.load(fp)
  model.load_state_dict(state_dict)

  ''' Data '''
  t = time()
  validset = ClassDiffDataset(args.data_path / args.dataset, split='valid')
  validloader = DataLoader(validset, batch_size=1, shuffle=False)
  print(f'load dataset: {time() - t:.3f}s')

  ''' Test '''
  model.eval()
  for X, Y in validloader:
    X = X.to(device)
    Y = Y.to(device)

    pred_A = model( X).argmax(axis=-1)[0]
    pred_B = model(-X).argmax(axis=-1)[0]

    print(f'{Y.item():02d}, {pred_A.item():02d}, {pred_B.item():02d}')


if __name__ == '__main__':
  parser = ArgumentParser()
  parser.add_argument('-D', '--dataset', default='cifar10')
  parser.add_argument('-R', '--n_ref',   default=3, type=int)
  parser.add_argument('--rdata_path', type=Path, default=Path('data'))
  parser.add_argument('--data_path',  type=Path, default=Path('preprocessed'))
  parser.add_argument('--log_path',   type=Path, default=Path('log'))
  args = parser.parse_args()

  args.exp_name = f'resnet18_{args.dataset}'

  test(args)
