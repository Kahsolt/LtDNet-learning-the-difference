#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/02/26 

from time import time
from pathlib import Path
from argparse import ArgumentParser

from tqdm import tqdm
import numpy as np
import torch
from torch import Tensor
from torch.autograd import grad
import torch.nn.functional as F
import torchvision.datasets as DS

from data import ClassSampleDataset
from model import ResNet18_32x32
from test import get_pred

device = 'cuda' if torch.cuda.is_available() else 'cpu'

img_dp = Path('adv')
img_dp.mkdir(exist_ok=True, parents=True)


@torch.enable_grad()
def pgd_diff(model, X:np.ndarray, Y:int, refset:ClassSampleDataset, steps=40, eps=8/255, alpha=1/255, method:str='fixed') -> Tensor:
  with torch.no_grad():
    X: Tensor = torch.from_numpy(X.copy())            # uint8, [B=1, C=3, H, W]
    X = X / 255.0
    X = X.to(device)
    AX = X.detach().clone()

    if method == 'fixed':
      R = refset.get_samples_of_class(Y, args.n_ref)  # uint8, [N, C=3, H, W]
      R: Tensor = torch.from_numpy(R.copy())
      R = R / 255.0
      R = R.detach().clone()
      R = R.to(device)

    tgt = torch.LongTensor([Y*refset.num_classes+Y])
    tgt = tgt.repeat(args.n_ref)
    tgt = tgt.to(device)

  for i in range(steps):
    if method == 'random':
      R = refset.get_samples_of_class(Y, args.n_ref)  # uint8, [N, C=3, H, W]
      R: Tensor = torch.from_numpy(R.copy())
      R = R / 255.0
      R = R.detach().clone()
      R = R.to(device)

    AX.requires_grad = True
    R .requires_grad = False

    D = (AX - R) / 4
    logits = model(D)
    loss = F.cross_entropy(logits, tgt, reduction='none')
    g = grad(loss, AX, loss)[0]

    AX = AX.detach() + g.sign() * alpha
    AX = X + (AX - X).clamp(-eps, eps)
    AX = AX.clamp(0.0, 1.0).detach()

    with torch.no_grad():
      if i % 5 == 0:
        l = loss.mean().item()
        print(f'>> loss: {l:.7f}')

      AX_i = (AX * 255.0).byte().cpu().numpy()
      pred = get_pred(model, AX_i, refset, args.n_ref)
      if pred != Y: break

  return AX.cpu()


@torch.no_grad()
def attack(args):
  print('>> [Attack]')

  ''' Model '''
  model = ResNet18_32x32(num_classes=100, pretrained=False).to(device)

  fp: Path = args.log_path / args.exp_name / 'model-best.pth'
  state_dict = torch.load(fp)
  model.load_state_dict(state_dict)

  ''' Data '''
  t = time()
  #refset = ClassSampleDataset(args.data_path / args.dataset, split='train')
  refset = ClassSampleDataset(args.data_path / args.dataset, split='valid')
  testset = DS.CIFAR10(root=args.rdata_path, train=False, download=True)
  print(f'load dataset: {time() - t:.3f}s')

  ''' Test '''
  total, atk, ok = 0, 0, 0
  model.eval()
  for i, (X, Y) in enumerate(tqdm(testset, total=None if args.limit < 0 else args.limit)):
    X = np.asarray(X)               # uint8
    X = X.transpose([2, 0, 1])
    X = np.expand_dims(X, axis=0)

    AX = pgd_diff(model, X, Y, refset, eps=8/255, alpha=1/255, steps=40, method=args.method)
    AX_i = (AX * 255.0).byte().cpu().numpy()
    y_pred = get_pred(model, AX_i, refset, args.n_ref)

    if True:
      D = np.abs(AX_i.astype(np.int16) - X.astype(np.int16))
      print('ok' if Y == y_pred else 'atk', D.max(), D.min(), D.mean(), D.std())

    ok    += Y == y_pred
    atk   += Y != y_pred
    total += 1

    if args.limit > 0 and i > args.limit: break

  print(f'>> acc: {ok / total:.3%}')
  print(f'>> asr: {atk / total:.3%}')


if __name__ == '__main__':
  parser = ArgumentParser()
  parser.add_argument('-D', '--dataset', default='cifar10')
  parser.add_argument('-R', '--n_ref',   default=3, type=int)
  parser.add_argument('-M', '--method',  default='fixed', choices=['fixed', 'random'])
  parser.add_argument('--limit',      default=100, type=int)
  parser.add_argument('--rdata_path', type=Path, default=Path('data'))
  parser.add_argument('--data_path',  type=Path, default=Path('preprocessed'))
  parser.add_argument('--log_path',   type=Path, default=Path('log'))
  args = parser.parse_args()

  args.exp_name = f'resnet18_{args.dataset}'

  attack(args)
