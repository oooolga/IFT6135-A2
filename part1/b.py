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
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=100, metavar='N',
                    help='number of epochs to train (default: 10)')
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

def weight_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        torch.nn.init.xavier_uniform(m.weight.data)
        m.bias.data.fill_(0)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(784, 800)
        self.fc2 = nn.Linear(800, 800)
        self.do = nn.Dropout()
        self.fc = nn.Linear(800, 10)

        self.apply(weight_init)
        self.weights = [ self.fc1.weight, self.fc2.weight, self.fc.weight]

    def forward(self, x):
        tmp = F.relu(self.fc1(x))
        tmp = F.relu(self.fc2(tmp))
        tmp = self.do(tmp)
        tmp = self.fc(tmp)
        return F.log_softmax(tmp)

    def predict1(self, x, N):
        tmp = F.relu(self.fc1(x))
        tmp = F.relu(self.fc2(tmp))
        tmp = self.fc(tmp*0.5)
        return F.log_softmax(tmp)

    def predict2(self, x, N):
        tmp = F.relu(self.fc1(x))
        tmp = F.relu(self.fc2(tmp))
        pre_softmax = 0
        for _ in range(N):
            pre_softmax += 1.0/N * self.fc(self.do(tmp))
        return F.log_softmax(pre_softmax)

    def predict3(self, x, N):
        tmp = F.relu(self.fc1(x))
        tmp = F.relu(self.fc2(tmp))
        preds = 0
        for _ in range(N):
            pre_softmax = self.fc(self.do(tmp))
            preds += F.log_softmax(pre_softmax) * 1.0 / N
        return preds

model = Net()
if args.cuda:
    model.cuda()

optimizer = optim.SGD(model.parameters(), lr=0.02)



def test(N, mode):
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)

        if mode == 1:
            output = model.predict1(data.view(-1, 784), N)
        if mode == 2:
            output = model.predict2(data.view(-1, 784), N)
        if mode == 3:
            output = model.predict3(data.view(-1, 784), N)

        test_loss += F.nll_loss(output, target, size_average=False).data[0] # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).long().cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return 100. * correct / len(test_loader.dataset)

for epoch in range(1, args.epochs + 1):
    model.train()
    correct = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data.view(-1, 784))
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).long().cpu().sum()

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data[0]))

Ns = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
test_accs1 = []
test_accs2 = []
test_accs3 = []
for N in Ns:
    test_accs1.append(test(N, 1))
    test_accs2.append(test(N, 2))
    test_accs3.append(test(N, 3))

plt.figure()
plt1, = plt.plot(Ns, test_accs1)
plt2, = plt.plot(Ns, test_accs2)
plt3, = plt.plot(Ns, test_accs3)
plt.legend([plt1,plt2,plt3], ["mode1", "mode2", "mode3"])
plt.savefig("compare_prediction.png")
