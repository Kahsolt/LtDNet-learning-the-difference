#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/02/23 

from time import time
from pathlib import Path
from argparse import ArgumentParser
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
import torchvision.models as M
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR

from data import DataLoader, ClassDiffDataset

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def train(args):
  print('>> [Train]')

  log_dp: Path = args.log_path / args.exp_name
  log_dp.mkdir(exist_ok=True, parents=True)

  ''' Model '''
  model = M.resnet18(weights=M.ResNet18_Weights.DEFAULT).to(device)
  optimizer = Adam(model.parameters(), lr=args.lr)
  
  fp = log_dp / 'model-best.pth'
  if fp.exists():
    state_dict = torch.load(fp)
    model.load_state_dict(state_dict)
  fp_optim = fp.with_name(fp.name + '.optim')
  if fp_optim.exists():
    optimizer.load_state_dict(state_dict['optimizer'])
    best_acc = state_dict['acc']
    last_epoch = state_dict['epoch']
    step = state_dict['step']
  else:
    best_acc = 0.0
    last_epoch = -1
    step = 0

  scheduler = StepLR(optimizer=optimizer, step_size=10, gamma=0.25, last_epoch=last_epoch)

  ''' Data '''
  t = time()
  trainset = ClassDiffDataset(args.data_path / args.dataset, split='train')
  trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
  validset = ClassDiffDataset(args.data_path / args.dataset, split='valid')
  validloader = DataLoader(validset, batch_size=1, shuffle=False)
  print(f'load dataset: {time() - t:.3f}s')

  ''' Helper '''
  epoch = None    # fix closure NameError
  def save_ckpt(fp:Path):
    torch.save(model.state_dict(), fp)
    torch.save({
      'optimizer': optimizer.state_dict(),
      'epoch': epoch,
      'step': step,
      'loss': train_loss[-1],
      'acc': valid_acc[-1],
      'acc_train': train_acc[-1],
    }, fp.with_name(fp.name + '.optim'))

  def save_plot(fp:Path):
    fig, ax1 = plt.subplots()
    ax1.plot(valid_acc, c='r', alpha=0.75, label='acc tr.')
    ax1.plot(train_acc, c='g', alpha=0.75, label='acc va.')
    ax2 = ax1.twinx()
    ax2.plot(loss_list, c='b', alpha=0.75, label='loss')
    fig.legend()
    fig.suptitle(args.exp_name)
    fig.savefig(fp, dpi=400)

  ''' Train '''
  train_loss = []
  train_acc  = []
  valid_acc  = []
  for epoch in range(max(0, last_epoch), args.epochs):
    ''' train '''
    loss_list, acc_list = [], []
    total, ok = 0, 0

    model.train()
    for X, Y in trainloader:
      step += 1

      X = X.to(device)
      Y = Y.to(device)

      optimizer.zero_grad()
      output = model(X)
      loss = F.cross_entropy(output, Y)
      loss.backward()
      optimizer.step()

      with torch.no_grad():
        pred = output.argmax(axis=-1)
        total += len(Y)
        ok    += (Y == pred).sum().item()

      if step % 1000 == 0:
        loss_list.append(loss.item())
        acc_list.append(ok / total)
        print(f"[Epoch {epoch+1} / Step {step}] loss: {loss_list[-1]:.7f}, accuracy: {acc_list[-1]:.3%}")
 
    train_loss.append(sum(loss_list) / len(loss_list))
    train_acc .append(ok / total)
    print(f">> [Epoch {epoch+1}] Train - loss: {train_loss[-1]:.7f}, accuracy: {train_acc[-1]:.3%}")

    ''' valid '''
    model.eval()
    ok, total = 0, 0
    with torch.no_grad():
      for X, Y in validloader:
        X = X.to(device)
        Y = Y.to(device)

        output = model(X)
        pred = output.argmax(axis=-1)

        total += len(Y)
        ok    += (Y == pred).sum().item()

    valid_acc.append(ok / total)
    print(f">> [Epoch {epoch+1}] Valid - accuracy: {valid_acc[-1]:.3%}")

    scheduler.step(valid_acc[-1])

    if valid_acc[-1] > best_acc:
      print('save new best ...')
      save_ckpt(log_dp / 'model-best.pth')
      save_plot(log_dp / 'stats-best.png')

    save_plot(log_dp / f'stats-{epoch+1}.png')
    save_ckpt(log_dp / f'model-{epoch+1}.pth')

  save_ckpt(log_dp / 'model-final.pth')
  save_plot(log_dp / 'stats-final.png')


if __name__ == '__main__':
  parser = ArgumentParser()
  parser.add_argument('-D', '--dataset',    default='cifar10')
  parser.add_argument('-E', '--epochs',     default=50, type=int)
  parser.add_argument('-B', '--batch_size', default=128, type=int)
  parser.add_argument('--lr',               default=0.01, type=float)
  parser.add_argument('--data_path', type=Path, default=Path('preprocessed'))
  parser.add_argument('--log_path',  type=Path, default=Path('log'))
  args = parser.parse_args()

  args.exp_name = f'resnet18_{args.dataset}'

  train(args)
