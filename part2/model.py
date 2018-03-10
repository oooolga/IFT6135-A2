__author__	= 	"Olga (Ge Ya) Xu"
__email__ 	=	"olga.xu823@gmail.com"

from util import *

class Net(nn.Module):

	def __init__(self):
		super(Net, self).__init__()

		in_dim = 64, 64, 3
		# output shape: n x 64 x 64 x 64
		conv1_kern_size, conv1_out_chan, conv1_stride, conv1_pad = 5, 64, 1, 2
		# output shape: n x 31 x 31 x 64
		pool1_size, pool1_stride = 3, 2
		# output shape: n x 31 x 31 x 128
		conv2_kern_size, conv2_out_chan, conv2_stride, conv2_pad = 5, 128, 1, 2
		# output shape: n x 15 x 15 x 128
		pool2_size, pool2_stride = 3, 2
		# output shape: n x 15 x 15 x 256
		conv3_kern_size, conv3_out_chan, conv3_stride, conv3_pad = 3, 256, 1, 1
		# output shape: n x 15 x 15 x 256
		conv4_kern_size, conv4_out_chan, conv4_stride, conv4_pad = 3, 256, 1, 1
		# output shape: n x 7 x 7 x 256
		pool4_size, pool4_stride = 3, 2

		mlp1_out = 4096
		mlp2_out = 4096
		mlp3_out = 1
		

		self.features = nn.Sequential(
			nn.Conv2d(in_dim[-1], conv1_out_chan, kernel_size=conv1_kern_size,
				stride=conv1_stride, padding=conv1_pad),
			nn.MaxPool2d(kernel_size=pool1_size, stride=pool1_stride),
			nn.BatchNorm2d(conv1_out_chan),
			nn.ReLU(inplace=True),
			nn.Conv2d(conv1_out_chan, conv2_out_chan, kernel_size=conv2_kern_size,
				stride=conv2_stride, padding=conv2_pad),
			nn.MaxPool2d(kernel_size=pool2_size, stride=pool2_stride),
			nn.Relu(inplace=True),
			nn.BatchNorm2d(conv2_out_chan),
			nn.Conv2d(conv2_out_chan, conv3_out_chan, kernel_size=conv3_kern_size,
				stride=conv3_stride, padding=conv3_pad),
			nn.Relu(inplace=True),
			nn.BatchNorm2d(conv3_out_chan),
			nn.Conv2d(conv3_out_chan, conv4_out_chan, kernel_size=conv4_kern_size,
				stride=conv4_stride, padding=conv4_pad),
			nn.MaxPool2d(kernel_size=pool4_size, stride=pool4_stride),
			nn.Relu(inplace=True),
			nn.BatchNorm2d(conv4_out_chan)
			)

		self.classifier = nn.Sequential(
			nn.Linear(7*7*256, mlp1_out, bias=False),
			nn.BatchNorm1d(mlp1_out),
			nn.ReLU(inplace=True),
			nn.Dropout(0.5),
			nn.Linear(mlp1_out, mlp2_out, bias=False),
			nn.BatchNorm1d(mlp2_out),
			nn.ReLU(inplace=True),
			nn.Dropout(0.5),
			nn.Linear(mlp2_out, mlp3_out)
			)


	def forward(self, x):
		f = self.features(x)
		f = f.view(-1, 7*7*256)
		c = self.classifier(f)
		return c
