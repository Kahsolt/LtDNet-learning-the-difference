#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/03/03

from time import time
from pathlib import Path
from argparse import ArgumentParser

import torch
import torchvision.models as M
import torchvision.datasets as DS
import numpy as np
from scipy import stats
from tqdm import tqdm

from data import ClassSampleDataset
from preprocess import get_d

device = 'cuda' if torch.cuda.is_available() else 'cpu'


@torch.inference_mode()
def test(args):
  print('>> [Test]')

  ''' Model '''
  model = M.resnet18().to(device)

  fp: Path = args.log_path / args.exp_name / 'model-best.pth'
  state_dict = torch.load(fp)
  model.load_state_dict(state_dict)

  ''' Data '''
  t = time()
  refset = ClassSampleDataset(args.data_path / args.dataset, split='train')
  #refset = ClassSampleDataset(args.data_path / args.dataset, split='valid')
  testset = DS.CIFAR10(root=args.rdata_path, train=False, download=True)
  print(f'load dataset: {time() - t:.3f}s')

  ''' Test '''
  NUM_CLASSES = refset.num_classes
  ok, total = 0, 0
  model.eval()
  for X, Y in tqdm(testset):
    X = np.asarray(X)               # uint8
    X = X.transpose([2, 0, 1])
    X = np.expand_dims(X, axis=0)

    RXs, XRs = [], []
    for cls in range(NUM_CLASSES):
      R = refset.get_samples_of_class(cls, args.n_ref)
      d = (get_d(R, X).astype(np.float32) + 0.5) / 127.5
      D = torch.from_numpy(d).to(device)
      pred_RX = model(D).argmax(axis=-1)
      RXs.extend([p % NUM_CLASSES for p in pred_RX.tolist()])
      
      R = refset.get_samples_of_class(cls, args.n_ref)
      d = (get_d(X, R).astype(np.float32) + 0.5) / 127.5
      D = torch.from_numpy(d).to(device)
      pred_XR = model(D).argmax(axis=-1)
      XRs.extend([p // NUM_CLASSES for p in pred_XR.tolist()])

    y_preds = RXs + XRs
    y_pred = stats.mode(y_preds, keepdims=False).mode

    total += 1
    ok += Y == y_pred

  print(f'>> accuracy: {ok / total:.3%}')


if __name__ == '__main__':
  parser = ArgumentParser()
  parser.add_argument('-D', '--dataset', default='cifar10')
  parser.add_argument('-K', '--n_ref',   default=3, type=int)
  parser.add_argument('--rdata_path', type=Path, default=Path('data'))
  parser.add_argument('--data_path',  type=Path, default=Path('preprocessed'))
  parser.add_argument('--log_path',   type=Path, default=Path('log'))
  args = parser.parse_args()

  args.exp_name = f'resnet18_{args.dataset}'

  test(args)
