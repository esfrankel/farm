import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from torch.distributions.bernoulli import Bernoulli
import torch.nn.functional as F
from constants import *


class Inference:

	def __init__(self, exp, ckpt):

		self.model = torch.load(Path('./checkpoints') / exp / ckpt)
		self.model = self.model.cuda()

		# load the dataset
		mnist = np.load(DATA_PATH)
		self.xtr, self.xte = mnist['train_data'], mnist['valid_data']
		self.xtr = torch.from_numpy(self.xtr).cuda() # [50000, 784]
		self.xte = torch.from_numpy(self.xte).cuda() # [10000, 784]

		self.model.eval()

		# plt.imshow(xtr[1000:1016].view(4, 28*4, 28).transpose(0,1).reshape(4*28, 4*28).cpu().numpy(), cmap='gray')
		# plt.show()

	def sample_single_order(self, B):
		'''Autoregressively samples a batch of size B with a single ordering.'''

		ordering = self.model.m[-1]

		x = torch.zeros((B, 784), dtype=torch.float32).cuda()
		for i in range(784):
			y = self.model(x)
			logits = y[:, ordering[i]]
			B = Bernoulli(torch.sigmoid(logits))
			sample = B.sample()
			x[:, ordering[i]] = sample

		return x

# TODO:
# - Autoregressive sampling
# - Fixed orders
# - Occlusion fixing tests

if __name__ == '__main__':
	I = Inference('made_made_natural_ordering', '099_params.pt')
	# import pdb; pdb.set_trace()
	x = I.sample_single_order(16)
	vis = x.view(4, 28*4, 28).transpose(0,1).reshape(4*28, 4*28)
	plt.imshow(vis.cpu().numpy(), cmap='gray')
	plt.show()