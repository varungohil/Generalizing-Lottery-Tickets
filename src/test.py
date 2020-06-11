from utils import *
from dataloader import *
from model import *
from parser import *
import torch
import random
import torchvision
import torch.optim as optim


def test(model, dataloader, device, model_path):
	"""
	Function to print the fraction of pruned weights and test accuracy of a model

	Arguments
	---------
	model : the PyTorch neural netowrk architecture
	dataloader : PyTorch dataloader for loading the dataset
	device : Device(GPU/CPU) on which to perform computation
	model_path: Path to trained model whose accuracy needs to be evaluated

	Returns:
	None
	"""
	cpt = torch.load(model_path)
	model.load_state_dict(cpt['model_state_dict'])
	model.eval()
	model.to(device)

	zero_total = 0
	zeros = 0
	for name, params in model.named_parameters():
		if "weight" in name:
			weight_copy = params.data.abs().clone()
			zeros += weight_copy.numel() - weight_copy.nonzero().size(0)
			zero_total += weight_copy.numel()
	print(f"Fraction of weights pruned = {zeros}/{zero_total} = {zeros/zero_total}")

	correct = 0
	total = 0
	with torch.no_grad():
		for data in dataloader:
			inputs, labels = data[0].to(device), data[1].to(device)
			outputs = model(inputs)
			_, predicted = torch.max(outputs.data, 1)
			total += labels.size(0)
			correct += (predicted == labels).sum().item()
	print(f"Accuracy: {100 * correct / total}")

if __name__ == '__main__':

	#Parsers the command line arguments
	parser = args_parser_test()
	args = parser.parse_args()

	#Uses GPU is available
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	print(f'Using {device} device.')

	#Loads dataset
	dataloader = load_dataset(args.dataset, args.batch_size, False)

	#Checks number of classes to aa appropriate linear layer at end of model
	if args.dataset in ['cifar10', 'fashionmnist', 'svhn']:
		num_classes = 10
	elif args.dataset in ['cifar100']:
		num_classes = 100
	else:
		raise ValueError(args.dataset + " dataset not supported")

	#Loads model
	model = load_model(args.architecture, num_classes)

	test(model, dataloader, device, args.model_path)