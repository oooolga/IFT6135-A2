__author__	= 	"Olga (Ge Ya) Xu"
__email__ 	=	"olga.xu823@gmail.com"

from util import *

def parse():
	parser = argparse.ArgumentParser()
	parser.add_argument('-lr', '--learning_rate', default=5e-2, type=float,
						help='Learning rate')
	parser.add_argument('-m', '--momentum', default=0.2, type=float, help="Momentum")
	parser.add_argument('-s', '--seed', default=111, type=int, help='Random seed')
	parser.add_argument('--batch_size', default=64, type=int,
						help='Mini-batch size for training')
	parser.add_argument('--test_batch_size', default=200, type=int,
						help='Mini-batch size for testing')
	parser.add_argument('--epoch', default=13, type=int, help='Number of epochs')
	parser.add_argument('-o', '--optimizer', default='SGD', type=str, help='Optimizer')
	parser.add_argument('-n', '--model_name', default='model_1', type=str, help='Model name')

	args = parser.parse_args()
	return args

def output_arguments(args):
	print 'Model name:\t\t{}'.format(args.model_name)
	print 'Seed:\t\t\t{}'.format(args.seed)
	print 'Total epoch:\t\t{}'.format(args.epoch)
	print 'Batch size:\t\t{}'.format(args.batch_size)
	print 'Learning rate:\t\t{}'.format(args.learning_rate)
	print 'Optimizer:\t\t{}'.format(args.optimizer)


if __name__ == '__main__':

	args = parse()

	torch.manual_seed(args.seed)

	if use_cuda:
		torch.cuda.manual_seed_all(args.seed)

	random.seed(args.seed)

	print 'Loading data...'
	#seperate_data()
	train_loader, valid_loader, test_loader = load_data(batch_size=args.batch_size,
														test_batch_size=args.test_batch_size)

	print 'Loading model...\n'

	model = Net()
	#model = ResNet(BottleneckBlock, [3,4,6,4])

	output_arguments(args)

	if use_cuda:
		model.cuda()

	run(model, train_loader, valid_loader, test_loader,  args.model_name, args.epoch,
		args.learning_rate, args.optimizer, args.momentum)