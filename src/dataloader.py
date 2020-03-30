import torch
import torchvision
from torch.utils.data import Dataset
import torch.optim as optim
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms

def load_dataset(dataset, batch_size = 512, is_train_split=True, root='../datasets'):
	"""
	Loads the dataset loader object

	Arguments
	---------
	dataset : Name of dataset which has to be loaded
	batch_size : Batch size to be used 
	is_train_split : Boolean which when true, indicates that training dataset will be loaded

	Returns
	-------
	Pytorch Dataloader object
	"""
	if is_train_split:
		if dataset == 'cifar10':
			transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
			transform_augment = transforms.Compose([transforms.RandomHorizontalFlip(), transforms. RandomCrop(32, padding=4)])
			data_set = torchvision.datasets.CIFAR10(root=root, train=is_train_split, download=True, transform=transforms.Compose([transform_augment, transform]))
			data_loader = torch.utils.data.DataLoader(data_set, batch_size=batch_size, shuffle=True)
		elif dataset == 'cifar100':
			transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
			transform_augment = transforms.Compose([transforms.RandomHorizontalFlip(), transforms. RandomCrop(32, padding=4)])
			data_set = torchvision.datasets.CIFAR100(root=root, train=is_train_split, download=True, transform=transforms.Compose([transform_augment, transform]))
			data_loader = torch.utils.data.DataLoader(data_set, batch_size=batch_size, shuffle=True)
		elif dataset == 'svhn':
			if not is_train_split:
				svhn_split = 'test'
			else:
				svhn_split = 'train'
			transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
			transform_augment = transforms.Compose([transforms.RandomHorizontalFlip(), transforms. RandomCrop(32, padding=4)])
			data_set = torchvision.datasets.SVHN(root=root, split=svhn_split , download=True, transform=transforms.Compose([transform_augment, transform]))
			data_loader = torch.utils.data.DataLoader(data_set, batch_size=batch_size, shuffle=True)
		elif dataset == 'fashionmnist':
			transform = transforms.Compose([transforms.Grayscale(3),transforms.ToTensor(),transforms.Normalize((0.5,), (0.5,))])
			transform_augment = transforms.Compose([transforms.RandomHorizontalFlip(), transforms. RandomCrop(32, padding=4)])
			data_set = torchvision.datasets.FashionMNIST(root=root, train=is_train_split, download=True, transform=transforms.Compose([transform_augment, transform]))
			data_loader = torch.utils.data.DataLoader(data_set, batch_size=batch_size, shuffle=True, num_workers=2)
		elif dataset == 'cifar10a' or dataset == 'cifar10b':
			transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
			transform_augment = transforms.Compose([transforms.RandomHorizontalFlip(), transforms. RandomCrop(32, padding=4)])
			cifarset = torchvision.datasets.CIFAR10(root=root, train=is_train_split, download=True, transform=transforms.Compose([transform_augment, transform]))
			label_flag = {x:True for x in range(10)}
			cifarA = []
			cifarB = []
			for sample in cifarset:
				if label_flag[sample[-1]]:
					cifarA.append(sample)
					label_flag[sample[-1]] = False
				else:
					cifarB.append(sample)
					label_flag[sample[-1]] = True
			class DividedCifar10A(torch.utils.data.dataset.Dataset):
				def __init__(self):
					self.samples = cifarA
				def __len__(self):
					return len(self.samples)
				def __getitem__(self, index):
					return self.samples[index]
			class DividedCifar10B(torch.utils.data.dataset.Dataset):
				def __init__(self):
					self.samples = cifarB
				def __len__(self):
					return len(self.samples)
				def __getitem__(self, index):
					return self.samples[index]
			if dataset == 'cifar10a' :
				data_set = DividedCifar10A()
			else:
				data_set == DividedCifar10B()
			data_loader = torch.utils.data.DataLoader(data_set, batch_size=batch_size, shuffle=True)
		else:
			raise ValueError("Dataset not supported.")
	else:
		if dataset == 'cifar10':
			transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
			# transform_augment = transforms.Compose([transforms.RandomHorizontalFlip(), transforms. RandomCrop(32, padding=4)])
			data_set = torchvision.datasets.CIFAR10(root=root, train=is_train_split, download=True, transform=transform)
			data_loader = torch.utils.data.DataLoader(data_set, batch_size=batch_size, shuffle=True)
		elif dataset == 'cifar100':
			transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
			# transform_augment = transforms.Compose([transforms.RandomHorizontalFlip(), transforms. RandomCrop(32, padding=4)])
			data_set = torchvision.datasets.CIFAR100(root=root, train=is_train_split, download=True, transform=transform)
			data_loader = torch.utils.data.DataLoader(data_set, batch_size=batch_size, shuffle=True)
		elif dataset == 'svhn':
			if not is_train_split:
				svhn_split = 'test'
			else:
				svhn_split = 'train'
			transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
			# transform_augment = transforms.Compose([transforms.RandomHorizontalFlip(), transforms. RandomCrop(32, padding=4)])
			data_set = torchvision.datasets.SVHN(root=root, split=svhn_split , download=True, transform=transform)
			data_loader = torch.utils.data.DataLoader(data_set, batch_size=batch_size, shuffle=True)
		elif dataset == 'fashionmnist':
			transform = transforms.Compose([transforms.Grayscale(3),transforms.ToTensor(),transforms.Normalize((0.5,), (0.5,))])
			# transform_augment = transforms.Compose([transforms.RandomHorizontalFlip(), transforms. RandomCrop(32, padding=4)])
			data_set = torchvision.datasets.FashionMNIST(root=root, train=is_train_split, download=True, transform=transform)
			data_loader = torch.utils.data.DataLoader(data_set, batch_size=batch_size, shuffle=True, num_workers=2)
		elif dataset == 'cifar10a' or dataset == 'cifar10b':
			transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
			transform_augment = transforms.Compose([transforms.RandomHorizontalFlip(), transforms. RandomCrop(32, padding=4)])
			cifarset = torchvision.datasets.CIFAR10(root=root, train=is_train_split, download=True, transform=transforms.Compose([transform_augment, transform]))
			label_flag = {x:True for x in range(10)}
			cifarA = []
			cifarB = []
			for sample in cifarset:
				if label_flag[sample[-1]]:
					cifarA.append(sample)
					label_flag[sample[-1]] = False
				else:
					cifarB.append(sample)
					label_flag[sample[-1]] = True
			class DividedCifar10A(torch.utils.data.dataset.Dataset):
				def __init__(self):
					self.samples = cifarA
				def __len__(self):
					return len(self.samples)
				def __getitem__(self, index):
					return self.samples[index]
			class DividedCifar10B(torch.utils.data.dataset.Dataset):
				def __init__(self):
					self.samples = cifarB
				def __len__(self):
					return len(self.samples)
				def __getitem__(self, index):
					return self.samples[index]
			if dataset == 'cifar10a' :
				data_set = DividedCifar10A()
			else:
				data_set == DividedCifar10B()
			data_loader = torch.utils.data.DataLoader(data_set, batch_size=batch_size, shuffle=True)
		else:
			raise ValueError(dataset + " dataset not supported.")
	return data_loader
