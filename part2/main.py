__author__	= 	"Olga (Ge Ya) Xu"
__email__ 	=	"olga.xu823@gmail.com"

from util import *

def parse():
	parser = argparse.ArgumentParser()
	parser.add_argument('-lr', '--learning_rate', default=1e-2, type=float,
						help='Learning rate')
	parser.add_argument('-m', '--momentum', default=0.5, type=float, help="Momentum")
	parser.add_argument('-s', '--seed', default=123, type=int, help='Random seed')
	parser.add_argument('--batch_size', default=50, type=int,
						help='Mini-batch size for training')
	parser.add_argument('--test_batch_size', default=1000, type=int,
						help='Mini-batch size for testing')
	parser.add_argument('--epoch', default=10, type=int, help='Number of epochs')

	args = parser.parse_args()
	return args

if __name__ == '__main__':

	args = parse()

	torch.manual_seed(args.seed)

	if use_cuda:
		torch.cuda.manual_seed_all(args.seed)

	print 'Loading data...'
	#seperate_data()
	train_loader, valid_loader, test_loader = load_data()

	print 'Loading model...'
	model = Net()