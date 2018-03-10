__author__	= 	"Olga (Ge Ya) Xu"
__email__ 	=	"olga.xu823@gmail.com"

from util import *

def _train(model, train_loader, optimizer):

	model.train()

	for batch_idx, (data, target) in enumerate(train_loader):

		if use_cuda:
			data, target = data.cuda(), target.cuda()

		data, target = Variable(data, requires_grad=False), Variable(target, requires_grad=False)

		optimizer.zero_grad()
		output = model(data)

		loss = F.binary_cross_entropy(F.sigmoid(output), target.float())
		loss.backward()
		optimizer.step()

		print '| | iter: {}\tloss: {}'.format(batch_idx, loss.data[0])

	return loss


def run(model, train_loader, valid_loader, test_loader, total_epoch, lr, momentum):
	optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

	for epoch in range(1, total_epoch+1):

		print('| epoch: {}'.format(epoch))
		loss = _train(model, train_loader, optimizer)