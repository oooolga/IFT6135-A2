__author__	= 	"Olga (Ge Ya) Xu"
__email__ 	=	"olga.xu823@gmail.com"

import pdb
import numpy as np
import argparse, os
import copy, glob, math, random
import scipy.misc

import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.optim as optim

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

DATA_PATH = './datasets'
TRAIN_PATH_OLD = './datasets/train_64x64'
TEST_PATH_OLD = './datasets/valid_64x64'
TRAIN_PATH = './datasets/train'
TEST_PATH = './datasets/test'
RESULT_PATH = './result'

IMG_PATH = './datasets/PetImages'

use_cuda = torch.cuda.is_available()
GLOBAL_TEMP = None

def visualize_kernel(kernel_tensor, im_name='conv1_kernel.jpg', pad=1, im_scale=100.0):

	def factorization(n):
		from math import sqrt
		for i in range(int(sqrt(float(n))), 0, -1):
			if n % i == 0:
				if i == 1: print('Who would enter a prime number of filters')
				return i, int(n / i)

	# map tensor wight in [0,255]
	max_w = torch.max(kernel_tensor)
	min_w = torch.min(kernel_tensor)
	scale = torch.abs(max_w-min_w)
	kernel_tensor = (kernel_tensor - min_w) / scale * 255.0
	kernel_tensor = torch.ceil(kernel_tensor)

	# pad kernel
	p2d = (pad, pad, pad, pad)
	padded_kernel_tensor = F.pad(kernel_tensor, p2d, 'constant', 0)

	# get the shape of output
	grid_Y, grid_X = factorization(kernel_tensor.size()[0])
	Y, X = padded_kernel_tensor.size()[2], padded_kernel_tensor.size()[3]

	# reshape
	# (grid_Y*grid_X) x y_dim x x_dim x num_chann
	padded_kernel_tensor = padded_kernel_tensor.permute(0, 2, 3, 1)
	padded_kernel_tensor = padded_kernel_tensor.cpu().view(grid_X, grid_Y*Y, X, -1)
	padded_kernel_tensor = padded_kernel_tensor.permute(0, 2, 1, 3)
	#padded_kernel_tensor = padded_kernel_tensor.view(1, grid_X*X, grid_Y*Y, -1)

	# kernel in numpy
	kernel_im = np.uint8((padded_kernel_tensor.data).numpy()).reshape(grid_X*X,
																	   grid_Y*Y, -1)
	kernel_im = scipy.misc.imresize(kernel_im, im_scale)
	print 'Saving {}...'.format(os.path.join(RESULT_PATH, im_name))
	scipy.misc.imsave(os.path.join(RESULT_PATH, im_name), kernel_im)



def seperate_data():

	if not os.path.exists(TRAIN_PATH):
		os.makedirs(TRAIN_PATH)
		os.makedirs(os.path.join(TRAIN_PATH, 'cat'))
		os.makedirs(os.path.join(TRAIN_PATH, 'dog'))

		for file in glob.glob(os.path.join(TRAIN_PATH_OLD, 'Cat*.jpg')):
			file_name = os.path.basename(file)
			os.rename(file, os.path.join(TRAIN_PATH, 'cat', file_name))

		for file in glob.glob(os.path.join(TRAIN_PATH_OLD, 'Dog*.jpg')):
			file_name = os.path.basename(file)
			os.rename(file, os.path.join(TRAIN_PATH, 'dog', file_name))

	if not os.path.exists(TEST_PATH):
		os.makedirs(TEST_PATH)
		os.makedirs(os.path.join(TEST_PATH, 'cat'))
		os.makedirs(os.path.join(TEST_PATH, 'dog'))

		for file in glob.glob(os.path.join(TEST_PATH_OLD, 'Cat*.jpg')):
			file_name = os.path.basename(file)
			os.rename(file, os.path.join(TEST_PATH, 'cat', file_name))

		for file in glob.glob(os.path.join(TEST_PATH_OLD, 'Dog*.jpg')):
			file_name = os.path.basename(file)
			os.rename(file, os.path.join(TEST_PATH, 'dog', file_name))


def load_data(batch_size=64, test_batch_size=1000):

	train_data = dset.ImageFolder(root=TRAIN_PATH, transform=transforms.ToTensor())
	train_imgs = random.sample(train_data.imgs, len(train_data))
	train_data.imgs = train_imgs[:-2499]
	train_loader = torch.utils.data.DataLoader(train_data,
											   batch_size=batch_size, shuffle=True)

	valid_data = dset.ImageFolder(root=TRAIN_PATH, transform=transforms.ToTensor())
	valid_data.imgs = train_imgs[-2499:]
	valid_loader = torch.utils.data.DataLoader(valid_data,
											   batch_size=batch_size, shuffle=True)

	test_data = dset.ImageFolder(root=TEST_PATH, transform=transforms.ToTensor())
	test_loader = torch.utils.data.DataLoader(test_data,
											  batch_size=test_batch_size, shuffle=True)

	return train_loader, valid_loader, test_loader

from model import *
from train_valid import *
from main import *
	