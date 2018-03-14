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

def _evaluate_data_set(model, data_loader, get_top_misclassified=None):

	model.eval()
	total_loss, correct = 0, 0
	total_data, total_batch = 0, 0
	top_ims = None

	if get_top_misclassified:
		top_ims = np.empty([0,3,64,64])
		top_count = 0

	for batch_idx, (data, target) in enumerate(data_loader):

		if use_cuda:
			data, target = data.cuda(), target.cuda()

		data, target = Variable(data, volatile=True, requires_grad=False), \
					Variable(target, requires_grad=False)

		output = model(data)
		

		out_eval = F.sigmoid(output)

		target = target.view(-1, 1)
		total_loss += F.binary_cross_entropy(out_eval, target.float()).data[0]

		if get_top_misclassified and top_count < get_top_misclassified:
			_, top_index = torch.max(torch.abs(target.float()-out_eval),0)
			top_index = top_index.data[0]
			top_count += 1
			top_data = data[top_index].view(1,3,64,64).cpu()
			top_data = top_data.data.numpy()
			top_ims = np.concatenate((top_ims, top_data), 0)

		out_eval = out_eval.data-0.5

		predicted = torch.ceil(out_eval).int()
		target = target.int()

		correct += (predicted == target.data).sum()

		total_data += len(data)
		total_batch += 1

	avg_loss = total_loss / float(total_batch)
	accuracy = correct / float(total_data)

	return avg_loss, accuracy, top_ims


def run(model, train_loader, valid_loader, test_loader, model_name, 
		total_epoch, lr, opt, momentum, lr_decay=1e-5, weight_decay=5e-5):

	if opt == 'Adagrad':
		print 'Learning rate decay:\t{}'.format(lr_decay)
		print 'Weight decay:\t\t{}\n'.format(weight_decay)
		optimizer = optim.Adagrad(model.parameters(), lr=lr, lr_decay=lr_decay,
								  weight_decay=weight_decay)
	if opt == 'Adam':
		print 'Weight decay:\t\t{}\n'.format(weight_decay)
		optimizer = optim.Adam(model.parameters(), lr=lr,
							   weight_decay=weight_decay)
	else:
		print 'Momentum:\t\t{}\n'.format(momentum)
		optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

	train_acc, valid_acc, test_acc = [], [], []
	train_loss, valid_loss, test_loss = [], [], []

	print('| epoch: {}'.format(0))
	avg_loss, accuracy, misclassified = _evaluate_data_set(model, train_loader, 25)
	train_loss.append(avg_loss)
	train_acc.append(accuracy)
	print('| train loss: {:.4f}\ttrain acc: {:.4f}'.format(avg_loss, accuracy))
	avg_loss, accuracy, _ = _evaluate_data_set(model, valid_loader)
	valid_loss.append(avg_loss)
	valid_acc.append(accuracy)
	print('| valid loss: {:.4f}\tvalid acc: {:.4f}'.format(avg_loss, accuracy))
	avg_loss, accuracy, _ = _evaluate_data_set(model, test_loader)
	test_loss.append(avg_loss)
	test_acc.append(accuracy)
	print('| test loss: {:.4f}\ttest acc: {:.4f}'.format(avg_loss, accuracy))

	# visualize kernels
	misclassified = torch.ceil(torch.FloatTensor(misclassified).cuda() * 255.0)
	visualize_kernel(misclassified,
					 im_name='misclassified_epoch_{}.jpg'.format(0),
					 model_name=model_name, im_scale=1.0, rescale=False)
	visualize_kernel(model.features[1].weight,
					 im_name='conv1_kernel_epoch_{}.jpg'.format(0),
					 model_name=model_name)

	for epoch in range(1, total_epoch+1):

		print('| epoch: {}'.format(epoch))
		_ = _train(model, train_loader, optimizer, verbose=False)

		# visualize kernels
		visualize_kernel(model.features[1].weight,
						 im_name='conv1_kernel_epoch_{}.jpg'.format(epoch),
						 model_name=model_name)
		
		# output training status
		avg_loss, accuracy, _ = _evaluate_data_set(model, train_loader)
		train_loss.append(avg_loss)
		train_acc.append(accuracy)
		print('| train loss: {:.4f}\ttrain acc: {:.4f}'.format(avg_loss, accuracy))
		avg_loss, accuracy, _ = _evaluate_data_set(model, valid_loader)
		valid_loss.append(avg_loss)
		valid_acc.append(accuracy)
		print('| valid loss: {:.4f}\tvalid acc: {:.4f}'.format(avg_loss, accuracy))
		avg_loss, accuracy, _ = _evaluate_data_set(model, test_loader)
		test_loss.append(avg_loss)
		test_acc.append(accuracy)
		print('| test loss: {:.4f}\ttest acc: {:.4f}'.format(avg_loss, accuracy))

	plot_acc_loss(train_loss, train_acc, valid_loss, valid_acc, test_loss, test_acc,
				  model_name=model_name)

	_, _, misclassified = _evaluate_data_set(model, train_loader, 25)
	misclassified = torch.ceil(torch.FloatTensor(misclassified).cuda() * 255.0)
	visualize_kernel(misclassified,
					 im_name='misclassified_epoch_{}.jpg'.format(epoch),
					 model_name=model_name, im_scale=1.0, rescale=False)
	