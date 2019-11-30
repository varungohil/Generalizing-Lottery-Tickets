from utils import *
import torch
import random
import torchvision
import torch.optim as optim
import numpy as np


def get_20_percent(total):
    return 0.2*total


def get_weight_fractions():
	percent_20s = []
	for i in range(31):
	    percent_20s.append(get_20_percent(100 - sum(percent_20s)))
	weight_fractions = []
	for i in range(31):
	    weight_fractions.append(sum(percent_20s[:i]))
	return weight_fractions


def permute_masks(old_masks, seed = 0):
    """ Function to randomly permute the mask in a global manner.
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


def prune_iteratively(model, dataloader, architecture, optimizer_type, device, models_path, init_path, random, is_equal_classes):
	if architecture == "vgg19":
		num_epochs = 160
		lr_anneal_epochs = [80, 120]
	elif architecture == "resnet50":
		num_epochs = 90
		lr_anneal_epochs = [50, 65, 80]
	else:
		ValueError(architecture + " architecture not supported")

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
			ValueError(optimizer_type + " optimizer not supported")

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
									ValueError(architecture + " architecture not supported")

		model.to(device)

		for epoch in range(1, num_epochs+1):
			if epoch in lr_anneal_epochs:
				optimizer.param_groups[0]['lr'] /= 10

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

			if epoch == num_epochs:
				if pruning_iter != 0:
					layer_index = 0
					for name, params in model.named_parameters():
						if "weight" in name:
							params.data.mul_(masks[layer_index].to(device))
							layer_index += 1
				torch.save({'epoch': epoch,'model_state_dict': model.state_dict(),'optimizer_state_dict': optimizer.state_dict() },models_path + "/"+ str(pruning_iter) + "_" + str(epoch))
	print("Finished Iterative Pruning")


if __name__ == '__main__':
	parser = args_parser_iterprune()
	args = parser.parse_args()

	random.seed(args.seed)

	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	print(f'Using {device} device.')


	if args.source_dataset in ['cifar10', 'svhn', 'fashionmnist']:
		num_classes_source = 10
	elif args.source_dataset in ['cifar100']:
		num_classes_source = 100
	else:
		ValueError(args.source_dataset + " as a source dataset is not supported")

	if args.target_dataset in ['cifar10', 'svhn', 'fashionmnist']:
		num_classes_target = 10
	elif args.target_dataset in ['cifar100']:
		num_classes_target = 100
	else:
		ValueError(args.target_dataset + " as a target dataset is not supported")

	dataloader = load_dataset(args.target_dataset, args.batch_size, True)

	model = load_model(args.architecture, num_classes_target)

	if num_classes_source == num_classes_target:
		prune_iteratively(model, dataloader, args.architecture, args.optimizer, device, args.model_saving_path, args.init_path, args.random, True)
	else:
		prune_iteratively(model, dataloader, args.architecture, args.optimizer, device, args.model_saving_path, args.init_path, args.random, False)

