import torch
import numpy as np

def initialize_xavier_normal(layer):
	"""
	Function to initialize a layer by picking weights from a xavier normal distribution

	Arguments
	---------
	layer : The layer of the neural network

	Returns
	-------
	None
	"""
	if type(layer) == nn.Conv2d:
		torch.nn.init.xavier_normal_(layer.weight)
		layer.bias.data.fill_(0)

def get_20_percent(total):
	"""
	Argument
	--------
	total : The number whose 20 percent we need to calculate

	Returns
	-------
	20% of total

	"""
	return 0.2*total


def get_weight_fractions():
	"""
	Returns a list of numbers which represent the fraction of weights pruned after each pruning iteration
	"""
	percent_20s = []
	for i in range(31):
		percent_20s.append(get_20_percent(100 - sum(percent_20s)))
	weight_fractions = []
	for i in range(31):
		weight_fractions.append(sum(percent_20s[:i]))
	return weight_fractions


def permute_masks(old_masks):
	""" 
	Function to randomly permute the mask in a global manner.
	Arguments
	---------
	old_masks: List containing all the layer wise mask of the neural network, mandatory. No default.
	seed: Integer containing the random seed to use for reproducibility. Default is 0

	Returns
	-------
	new_masks: List containing all the masks permuted globally
	"""

	layer_wise_flatten = []                      # maintain the layerwise flattened tensor
	for i in range(len(old_masks)):
		layer_wise_flatten.append(old_masks[i].flatten())

	global_flatten = []
	for i in range(len(layer_wise_flatten)):
		if len(global_flatten) == 0:
			global_flatten.append(layer_wise_flatten[i].cpu())
		else:
			global_flatten[-1] = np.append(global_flatten[-1], layer_wise_flatten[i].cpu())
	permuted_mask = np.random.permutation(global_flatten[-1])

	new_masks = []
	idx1 = 0
	idx2 = 0
	for i in range(len(old_masks)):
		till_idx = old_masks[i].numel()
		idx2 = idx2 + till_idx
		new_masks.append(permuted_mask[idx1:idx2].reshape(old_masks[i].shape))
		idx1 = idx2

	# Convert to tensor
	for i in range(len(new_masks)):
		new_masks[i] = torch.tensor(new_masks[i])

	return new_masks
