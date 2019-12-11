import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from torch.distributions.bernoulli import Bernoulli
import torch.nn.functional as F
from constants import *
from util import *
from multiprocessing import Process, Manager
import multiprocessing as mp


class Inference:

	def __init__(self, exp, ckpt):
		self.model = torch.load(Path('./checkpoints') / exp / ckpt)
		self.model = self.model.cuda()
		self.model.eval()
        # flipped_orderings = []
		# for i in range(num_workers):
		# 	ordering = self.model.orderings[i]
		# 	flipped_order = flip_ordering(ordering)
		# 	flipped_orderings.append(flipped_order)
        #
        # self.flipped_orderings = flipped_orderings

    def fill_image(image, track, i):
        ordering = self.flipped_orderings[i] # chooses the ordering
        while np.sum(list(track)) < 784: # while it's not completely full
            index = ordering[0] # gets the index it needs to fill next
            while track[index] == 1: # while the current index it wants is filled
                if len(ordering) == 1: # if that's the last one, then let's quit
                    return

                ordering = ordering[1:] # remove the first element of the ordering list since it's filled
                index = ordering[0] # index gets updated

            y = self.model(np.asarray(image)) # get model for y
            logits = y[curr_fill_val]

            B = Bernoulli(torch.sigmoid(logits))
            sample = B.sample()
            image[curr_fill_val] = sample

            track[curr_fill_val] = 1

        return


	def sample_single_ordering(self, B):
		'''Autoregressively samples a batch of size B with a single ordering.'''

		flip_order = [i for _, i in sorted(zip(self.model.m[-1], list(range(784))))]

		x = torch.zeros((B, 784), dtype=torch.float32).cuda()
		for i in range(784):
			y = 0
			for j in range(self.model.num_masks): # average over all masks
				self.model.update_masks(resample_hidden_masks=True, resample_ordering=False)
				y += self.model(x)
			y /= self.model.num_masks
			logits = y[:, flip_order[i]]
			B = Bernoulli(torch.sigmoid(logits))
			sample = B.sample()
			x[:, flip_order[i]] = sample

		return x


	def fill_occlusion(self, inp, occ):
		'''
		Fill occlusion over all images in inp using the best possible ordering.

		inp (torch.Tensor): (B, 784) collection of images to run on.
		occ (torch.Tensor): (784,) occlusion mask.  1 means occluded, 0 means not.
		'''

		x = inp * (1 - occ)
		best_ordering = None
		best_score = -1
		for i in range(len(self.model.orderings)):
			ordering = self.model.orderings[i]
			flip_order = flip_ordering(ordering)
			count = 0
			while occ[flip_order[count]] == 0:
				count += 1
			# print(i, count)
			if count > best_score:
				# print("Hi!")
				best_score = count
				best_ordering = ordering
				best_i = i

		# print('Using ordering ', best_i)

		self.model.update_masks(use_ordering=best_ordering)

		flip_order = flip_ordering(best_ordering)
		for i in range(784):
			if occ[flip_order[i]] == 0:
				continue # we only want to sample if the pixel is occluded
			y = 0
			for j in range(self.model.num_masks): # average over all masks
				self.model.update_masks(resample_hidden_masks=True, resample_ordering=False)
				y += self.model(x)
			y /= self.model.num_masks
			logits = y[:, flip_order[i]]
			B = Bernoulli(torch.sigmoid(logits))
			sample = B.sample()
			x[:, flip_order[i]] = sample

		return x

	def create_image_multiple_orderings(self):
		'''
		B (number): batch size B.
		'''
		mp.set_start_method('spawn')
		num_workers = len(self.model.orderings)
		flipped_orderings = []
		for i in range(num_workers):
			ordering = self.model.orderings[i]
			flipped_order = flip_ordering(ordering)
			flipped_orderings.append(flipped_order)

		manager = Manager()
		image = np.zeros(784)
		track = np.zeros(784)
		#image = manager.list(range = 784) # image that we are generating
		#track = manager.list(range = 784) # tracks what points of the image are filled

		pool = []
		for i in range(num_workers):
			p = Process(target = fill_image, args = (image, track, i))
			p.start()
			pool.append(p)

		for p in pool:
			p.join()

		image = torch.from_numpy(np.asarray(list(image)))
		return image

if __name__ == '__main__':
	I = Inference('made_more_hiddens_8_orderings', '059_params.pt')
	# import pdb; pdb.set_trace()
	x = I.sample_single_ordering(16)
	vis = x.view(4, 28*4, 28).transpose(0,1).reshape(4*28, 4*28)
	plt.imshow(vis.cpu().numpy(), cmap='gray')
	plt.show()
