__author__	= 	"Olga (Ge Ya) Xu"
__email__ 	=	"olga.xu823@gmail.com"

import pdb
import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
import argparse, os
import copy, glob, math

import random

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

DATA_PATH = './datasets'
TRAIN_PATH_OLD = './datasets/train_64x64'
TEST_PATH_OLD = './datasets/valid_64x64'
TRAIN_PATH = './datasets/train'
TEST_PATH = './datasets/test'

IMG_PATH = './datasets/PetImages'

use_cuda = torch.cuda.is_available()
GLOBAL_TEMP = None

from model import *
from train_valid import *

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
	