# -*- coding: utf-8 -*-

from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from models import Net
from torchvision import datasets, transforms
from torch.autograd import Variable
from tools import utils

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--save-interval', type=int, default=1, metavar='N',
                    help='how many epochs to wait before saving training status')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)  # 为CPU设置种子用于生成随机数，以使得结果是确定的
if args.cuda:
    # 为当前GPU设置随机种子；如果使用多个GPU，应该使用torch.cuda.manual_seed_all()为所有的GPU设置种子。
    torch.cuda.manual_seed(args.seed)


kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
"""加载数据。组合数据集和采样器，提供数据上的单或多进程迭代器
参数：
dataset：Dataset类型，从其中加载数据
batch_size：int，可选。每个batch加载多少样本
shuffle：bool，可选。为True时表示每个epoch都对数据进行洗牌
sampler：Sampler，可选。从数据集中采样样本的方法。
num_workers：int，可选。加载数据时使用多少子进程。默认值为0，表示在主进程中加载数据。
collate_fn：callable，可选。
pin_memory：bool，可选
drop_last：bool，可选。True表示如果最后剩下不完全的batch,丢弃。False表示不丢弃。
"""


train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('datas', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('datas', train=False, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])),
    batch_size=args.batch_size, shuffle=True, **kwargs)


model = Net()
if args.cuda:
    model.cuda()  # 将所有的模型参数移动到GPU上

optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)


def train(epoch):
    model.train()  # 把module设成training模式，对Dropout和BatchNorm有影响
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        # Variable类对Tensor对象进行封装，会保存该张量对应的梯度，以及对生成该张量的函数grad_fn的一个引用。如果该张量是用户创建的，grad_fn是None，称这样的Variable为叶子Variable。
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)  # 负log似然损失
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data[0]))


def test(epoch):
    model.eval()  # 把module设置为评估模式，只对Dropout和BatchNorm模块有影响
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        test_loss += F.nll_loss(output, target).data[0]  # Variable.data
        # get the index of the max log-probability
        pred = output.data.max(1)[1]
        correct += pred.eq(target.data).cpu().sum()

    test_loss = test_loss
    # loss function already averages over batch size
    test_loss /= len(test_loader)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


for epoch in range(1, args.epochs + 1):
    train(epoch)
    test(epoch)
    if ((epoch + 1) % args.save_interval) == 0:
        path_checkpoint = '{0}/{1}_state_epoch{2}.pth'.format(
            'checkpoints', 'mnist', epoch + 1)
        utils.save_checkpoint(model.state_dict(), path_checkpoint)
