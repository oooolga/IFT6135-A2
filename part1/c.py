from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import ipdb

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=100, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--use-bn', action='store_true')
parser.add_argument('--log-interval', default=10)

args = parser.parse_args()
args.cuda = torch.cuda.is_available()


kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=args.batch_size, shuffle=True, **kwargs)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        if not args.use_bn:
            self.conv = nn.Sequential(
                    nn.Conv2d(in_channels=1,
                        out_channels=16,
                        kernel_size=(3, 3),
                        padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=(2,2),stride=2),

                    nn.Conv2d(in_channels=16,
                        out_channels=32,
                        kernel_size=(3,3),
                        padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=(2,2),
                        stride=2),

                    nn.Conv2d(in_channels=32,
                        out_channels=64,
                        kernel_size=(3,3),
                        padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=(2,2),
                        stride=2),

                    nn.Conv2d(in_channels=64,
                        out_channels=128,
                        kernel_size=(3,3),
                        padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=(2,2),
                        stride=2)
                    )
        else:
            self.conv = nn.Sequential(
                    nn.Conv2d(in_channels=1,
                        out_channels=16,
                        kernel_size=(3, 3),
                        padding=1),
                    nn.BatchNorm2d(16),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=(2,2),stride=2),

                    nn.Conv2d(in_channels=16,
                        out_channels=32,
                        kernel_size=(3,3),
                        padding=1),
                    nn.BatchNorm2d(32),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=(2,2),
                        stride=2),

                    nn.Conv2d(in_channels=32,
                        out_channels=64,
                        kernel_size=(3,3),
                        padding=1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=(2,2),
                        stride=2),

                    nn.Conv2d(in_channels=64,
                        out_channels=128,
                        kernel_size=(3,3),
                        padding=1),
                    nn.BatchNorm2d(128),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=(2,2),
                        stride=2)
                    )
        self.clf = nn.Linear(128, 10)

    def forward(self, x):
        tmp = self.clf(self.conv(x).squeeze())
        return F.log_softmax(tmp)

model = Net()
if args.cuda:
    model.cuda()


optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)


def test():
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        test_loss += F.nll_loss(output, target, size_average=False).data[0] # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).long().cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return 100. * correct / len(test_loader.dataset)

train_accs = []
test_accs = []
for epoch in range(1, args.epochs + 1):
    model.train()
    correct = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).long().cpu().sum()

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data[0]))
    train_accs.append(100. * correct / len(train_loader.dataset))
    test_accs.append(test())

plt.figure()
tr_plt, = plt.plot(train_accs)
te_plt, = plt.plot(test_accs)
plt.legend([tr_plt, te_plt], ['train_acc', 'test_acc'])
plt.savefig('c_accs.png')
