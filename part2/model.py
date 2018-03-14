__author__	= 	"Olga (Ge Ya) Xu"
__email__ 	=	"olga.xu823@gmail.com"

from util import *

class Net(nn.Module):

	def __init__(self):
		super(Net, self).__init__()

		in_dim = 64, 64, 3
		# output shape: n x 64 x 64 x 32
		conv1_kern_size, conv1_out_chan, conv1_stride, conv1_pad = 5, 32, 1, 2
		# output shape: n x 31 x 31 x 32
		pool1_size, pool1_stride = 3, 2
		# output shape: n x 31 x 31 x 64
		conv2_kern_size, conv2_out_chan, conv2_stride, conv2_pad = 5, 64, 1, 2
		# output shape: n x 15 x 15 x 64
		pool2_size, pool2_stride = 3, 2
		# output shape: n x 15 x 15 x 128
		conv3_kern_size, conv3_out_chan, conv3_stride, conv3_pad = 3, 128, 1, 1
		# output shape: n x 15 x 15 x 128
		conv4_kern_size, conv4_out_chan, conv4_stride, conv4_pad = 3, 128, 1, 1
		# output shape: n x 7 x 7 x 128
		pool4_size, pool4_stride = 3, 2

		mlp1_out = 2048
		mlp2_out = 2048
		mlp3_out = 1
		

		self.features = nn.Sequential(
			nn.BatchNorm2d(in_dim[-1]),
			nn.Conv2d(in_dim[-1], conv1_out_chan, kernel_size=conv1_kern_size,
				stride=conv1_stride, padding=conv1_pad),
			nn.MaxPool2d(kernel_size=pool1_size, stride=pool1_stride),
			nn.ReLU(inplace=True),
			nn.BatchNorm2d(conv1_out_chan),
			nn.Conv2d(conv1_out_chan, conv2_out_chan, kernel_size=conv2_kern_size,
				stride=conv2_stride, padding=conv2_pad),
			nn.MaxPool2d(kernel_size=pool2_size, stride=pool2_stride),
			nn.ReLU(inplace=True),
			nn.BatchNorm2d(conv2_out_chan),
			nn.Conv2d(conv2_out_chan, conv3_out_chan, kernel_size=conv3_kern_size,
				stride=conv3_stride, padding=conv3_pad),
			nn.ReLU(inplace=True),
			nn.BatchNorm2d(conv3_out_chan),
			nn.Conv2d(conv3_out_chan, conv4_out_chan, kernel_size=conv4_kern_size,
				stride=conv4_stride, padding=conv4_pad),
			nn.MaxPool2d(kernel_size=pool4_size, stride=pool4_stride),
			nn.ReLU(inplace=True),
			nn.BatchNorm2d(conv4_out_chan)
			)

		self.classifier = nn.Sequential(
			nn.Linear(7*7*conv4_out_chan, mlp1_out, bias=False),
			nn.BatchNorm1d(mlp1_out),
			nn.ReLU(inplace=True),
			nn.Dropout(0.5),
			nn.Linear(mlp1_out, mlp2_out, bias=False),
			nn.BatchNorm1d(mlp2_out),
			nn.ReLU(inplace=True),
			nn.Dropout(0.5),
			nn.Linear(mlp2_out, mlp3_out)
			)

		self.out_chan = conv4_out_chan


	def forward(self, x):
		self.f = self.features(x)
		f = self.f.view(-1, 7*7*self.out_chan)
		c = self.classifier(f)
		return c

class BottleneckBlock(nn.Module):
	expansion = 4

	def __init__(self, in_chan, out_chan, stride=1, downsample=None):

		super(BottleneckBlock, self).__init__()

		self.conv1 = nn.Conv2d(in_chan, out_chan, kernel_size=1, bias=False)
		self.bn1 = nn.BatchNorm2d(out_chan)
		self.conv2 = nn.Conv2d(out_chan, out_chan, kernel_size=3, stride=stride,
							   padding=1, bias=False)
		self.bn2 = nn.BatchNorm2d(out_chan)
		self.conv3 = nn.Conv2d(out_chan, out_chan * 4, kernel_size=1, bias=False)
		self.bn3 = nn.BatchNorm2d(out_chan * 4)
		self.relu = nn.ReLU(inplace=True)
		self.downsample = downsample
		self.stride = stride

	def forward(self, x):
		residual = x

		conv1 = self.conv1(x)
		conv1 = self.bn1(conv1)
		conv1 = self.relu(conv1)

		conv2 = self.conv2(conv1)
		conv2 = self.bn2(conv2)
		conv2 = self.relu(conv2)

		conv3 = self.conv3(conv2)
		conv3 = self.bn3(conv3)

		if self.downsample is not None:
			residual = self.downsample(x)

		out = conv3 + residual
		out = self.relu(out)

		return out

class ResNet(nn.Module):

	def __init__(self, block, layers, num_classes=1):
		self.in_chan = 64
		super(ResNet, self).__init__()
		self.conv1 = nn.Conv2d(3, self.in_chan, kernel_size=5, stride=1, padding=3,
							   bias=False)
		self.bn1 = nn.BatchNorm2d(self.in_chan)
		self.relu = nn.ReLU(inplace=True)
		self.maxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
		self.layer1 = self._make_layer(block, 64, layers[0])
		self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
		self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
		self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
		self.avgpool = nn.AvgPool2d(7, stride=1)
		self.fc = nn.Linear(4608 * block.expansion, num_classes)

		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
				m.weight.data.normal_(0, math.sqrt(2. / n))
			elif isinstance(m, nn.BatchNorm2d):
				m.weight.data.fill_(1)
				m.bias.data.zero_()

	def _make_layer(self, block, out_chan, blocks, stride=1):

		downsample = None
		if stride != 1 or self.in_chan != out_chan * block.expansion:
			downsample = nn.Sequential(
				nn.Conv2d(self.in_chan, out_chan * block.expansion,
						  kernel_size=1, stride=stride, bias=False),
				nn.BatchNorm2d(out_chan * block.expansion),
			)

		layers = []
		layers.append(block(self.in_chan, out_chan, stride, downsample))
		self.in_chan = out_chan * block.expansion
		for i in range(1, blocks):
			layers.append(block(self.in_chan, out_chan))

		return nn.Sequential(*layers)

	def forward(self, x):
		x = self.conv1(x)
		x = self.bn1(x)
		x = self.relu(x)
		x = self.maxpool(x)
		x = self.layer1(x)
		x = self.layer2(x)
		x = self.layer3(x)
		x = self.layer4(x)

		x = self.avgpool(x)
		x = x.view(x.size(0), -1)
		x = self.fc(x)

		return x


