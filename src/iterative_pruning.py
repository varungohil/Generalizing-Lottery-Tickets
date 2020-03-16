from utils import *
from dataloader import *
from model import *
from parser import *
import torch
import random
import torchvision
import torch.optim as optim
import numpy as np
from tqdm import tqdm 

def prune_iteratively(model, dataloader, architecture, optimizer_type, device, models_path, init_path, random, is_equal_classes):
	"""
	Performs iterative pruning

	Arguments
	---------
	model : the PyTorch neural network model to be trained
	dataloader : PyTorch dataloader for loading the dataset
	architecture : The neural network architecture (VGG19 or ResNet50)
	optimizer_type : The optimizer to use for training (SGD / Adam)
	device : Device(GPU/CPU) on which to perform computation
	models_path: Path to directory where trained model/checkpoints will be saved
	init_path : Path to winning ticket initialization model
	random    : Boolean which when True perform pruning for random ticket
	is_equal_classes : Boolean to indicate is source and target dataset have equal number of classes

	Returns
	--------
	None
	"""
	if architecture == "vgg19":
		num_epochs = 160
		lr_anneal_epochs = [80, 120]
	elif architecture == "resnet50":
		num_epochs = 90
		lr_anneal_epochs = [50, 65, 80]
	else:
		raise ValueError(architecture + " architecture not supported")

	criterion = nn.CrossEntropyLoss().cuda()

	weight_fractions = get_weight_fractions()

	print("Iterative Pruning started")
	for pruning_iter in range(0,31):
		print(f"Running pruning iteration {pruning_iter}")
		if optimizer_type == 'sgd':
			optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0001)
		elif optimizer_type == 'adam':
			optimizer = optim.Adam(model.parameters(), lr=0.0003, weight_decay=0.0001)
		else:
			raise ValueError(optimizer_type + " optimizer not supported")

		if pruning_iter != 0:
			cpt = torch.load(models_path + f"/{pruning_iter-1}_{num_epochs}")
			model.load_state_dict(cpt['model_state_dict'])

			masks = []
			flat_model_weights = np.array([])
			for name, params in model.named_parameters():
				if "weight" in name:
					layer_weights = params.data.cpu().numpy()
					flat_model_weights = np.concatenate((flat_model_weights, layer_weights.flatten()))
			threshold = np.percentile(abs(flat_model_weights), weight_fractions[pruning_iter])

			zeros = 0
			total = 0
			for name, params in model.named_parameters():
				if "weight" in name:
					weight_copy = params.data.abs().clone()
					mask = weight_copy.gt(threshold).float()
					zeros += mask.numel() - mask.nonzero().size(0)
					total += mask.numel()
					masks.append(mask)
					if random != 'false':
						masks = permute_masks(masks)
			print(f"Fraction of weights pruned = {zeros}/{total} = {zeros/total}")  

		if random == 'false':
			if is_equal_classes:
				cpt = torch.load(init_path)
				model.load_state_dict(cpt['model_state_dict'])
			else:
				cpt = torch.load(init_path)
				new_dict = model.state_dict()
				for key in new_dict.keys():
					if "classifier" not in key and "fc" not in key:
						new_dict[key] = cpt['model_state_dict'][key]
						model.load_state_dict(new_dict)
					else:
						for m in model.modules():
							if isinstance(model, nn.Conv2d):
								if architecture == 'vgg19':
									nn.init.xavier_normal_(m.weight)
									layer.bias.data.fill_(0)
								elif architecture == 'resnet50':
									nn.init.kaiming_normal_(m.weight)
								else:
									raise ValueError(architecture + " architecture not supported")

		model.to(device)

		pruning_cycle = tqdm(range(1, num_epochs+1))
		for epoch in pruning_cycle:
			if epoch in lr_anneal_epochs:
				optimizer.param_groups[0]['lr'] /= 10

			correct = 0
			total = 0
			for batch_num, data in enumerate(dataloader, 0):
				inputs, labels = data[0].to(device), data[1].to(device)
				optimizer.zero_grad()
				
				if pruning_iter != 0:
					layer_index = 0
					for name, params in model.named_parameters():
						if "weight" in name:
							params.data.mul_(masks[layer_index].to(device))
							layer_index += 1

				outputs = model(inputs)
				loss = criterion(outputs, labels)
				loss.backward()
				optimizer.step()

				_, predicted = torch.max(outputs.data, 1)
				total += labels.size(0)
				correct += (predicted == labels).sum().item()

			if epoch == num_epochs:
				if pruning_iter != 0:
					layer_index = 0
					for name, params in model.named_parameters():
						if "weight" in name:
							params.data.mul_(masks[layer_index].to(device))
							layer_index += 1
				torch.save({'epoch': epoch,'model_state_dict': model.state_dict(),'optimizer_state_dict': optimizer.state_dict() },models_path + "/"+ str(pruning_iter) + "_" + str(epoch))
			pruning_cycle.set_description(f"Accuracy: {100 * correct / total}")
	print("Finished Iterative Pruning")


if __name__ == '__main__':
	#Parsers the command line arguments
	parser = args_parser_iterprune()
	args = parser.parse_args()

	#Sets random seed
	random.seed(args.seed)

	#Uses GPU is available
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	print(f'Using {device} device.')


	#Checks number of classes to aa appropriate linear layer at end of model
	if args.source_dataset in ['cifar10', 'svhn', 'fashionmnist']:
		num_classes_source = 10
	elif args.source_dataset in ['cifar100']:
		num_classes_source = 100
	else:
		raise ValueError(args.source_dataset + " as a source dataset is not supported")

	if args.target_dataset in ['cifar10', 'svhn', 'fashionmnist']:
		num_classes_target = 10
	elif args.target_dataset in ['cifar100']:
		num_classes_target = 100
	else:
		raise ValueError(args.target_dataset + " as a target dataset is not supported")

	#Loads dataset
	dataloader = load_dataset(args.target_dataset, args.batch_size, True)

	#Loads model
	model = load_model(args.architecture, num_classes_target)

	if num_classes_source == num_classes_target:
		prune_iteratively(model, dataloader, args.architecture, args.optimizer, device, args.model_saving_path, args.init_path, args.random, True)
	else:
		prune_iteratively(model, dataloader, args.architecture, args.optimizer, device, args.model_saving_path, args.init_path, args.random, False)

