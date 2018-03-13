__author__	= 	"Olga (Ge Ya) Xu"
__email__ 	=	"olga.xu823@gmail.com"

from util import *

def _train(model, train_loader, optimizer, verbose):

	model.train()

	for batch_idx, (data, target) in enumerate(train_loader):

		if use_cuda:
			data, target = data.cuda(), target.cuda()

		data, target = Variable(data, requires_grad=False), Variable(target, requires_grad=False)

		optimizer.zero_grad()
		output = model(data)

		target = target.view(-1, 1)
		loss = F.binary_cross_entropy(F.sigmoid(output), target.float())
		loss.backward()
		optimizer.step()

		if verbose and batch_idx % 50 == 0:
			print '| | iter: {}\tloss: {:.4f}'.format(batch_idx, loss.data[0])

	return loss

def _evaluate_data_set(model, data_loader):

	model.eval()
	total_loss, correct = 0, 0
	total_data, total_batch = 0, 0

	for batch_idx, (data, target) in enumerate(data_loader):

		if use_cuda:
			data, target = data.cuda(), target.cuda()

		data, target = Variable(data, volatile=True, requires_grad=False), \
					Variable(target, requires_grad=False)

		output = model(data)

		out_eval = F.sigmoid(output)

		target = target.view(-1, 1)
		total_loss += F.binary_cross_entropy(out_eval, target.float()).data[0]

		out_eval = out_eval.data-0.5

		predicted = torch.ceil(out_eval).int()
		target = target.int()

		correct += (predicted == target.data).sum()

		total_data += len(data)
		total_batch += 1

	avg_loss = total_loss / float(total_batch)
	accuracy = correct / float(total_data)
	return avg_loss, accuracy


def run(model, train_loader, valid_loader, test_loader, total_epoch, lr, opt, momentum,
		lr_decay=1e-5):

	if opt == 'Adagrad':
		print 'Learning rate decay:\t{}\n'.format(lr_decay)
		optimizer = optim.Adagrad(model.parameters(), lr=lr, lr_decay=lr_decay)
	else:
		print 'Momentum:\t\t{}\n'.format(momentum)
		optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

	print('| epoch: {}'.format(0))
	avg_loss, accuracy = _evaluate_data_set(model, train_loader)
	print('| train loss: {:.4f}\ttrain acc: {:.4f}'.format(avg_loss, accuracy))
	avg_loss, accuracy = _evaluate_data_set(model, valid_loader)
	print('| valid loss: {:.4f}\tvalid acc: {:.4f}'.format(avg_loss, accuracy))
	avg_loss, accuracy = _evaluate_data_set(model, test_loader)
	print('| test loss: {:.4f}\ttest acc: {:.4f}'.format(avg_loss, accuracy))

	for epoch in range(1, total_epoch+1):

		print('| epoch: {}'.format(epoch))
		_ = _train(model, train_loader, optimizer, verbose=True)

		# visualize kernels
		visualize_kernel(model.features[1].weight)
		
		# output training status
		avg_loss, accuracy = _evaluate_data_set(model, train_loader)
		print('| train loss: {:.4f}\ttrain acc: {:.4f}'.format(avg_loss, accuracy))
		avg_loss, accuracy = _evaluate_data_set(model, valid_loader)
		print('| valid loss: {:.4f}\tvalid acc: {:.4f}'.format(avg_loss, accuracy))
		avg_loss, accuracy = _evaluate_data_set(model, test_loader)
		print('| test loss: {:.4f}\ttest acc: {:.4f}'.format(avg_loss, accuracy))