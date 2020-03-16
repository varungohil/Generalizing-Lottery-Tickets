import argparse

def args_parser_train():
	"""
	returns argument parser object used while training a model
	"""
	parser = argparse.ArgumentParser()
	parser.add_argument('--architecture', required=True, type=str, help='neural network architecture [vgg19, resnet50]') 
	parser.add_argument('--dataset',type=str, required=True, help='dataset [cifar10, cifar100, svhn, fashionmnist]')
	parser.add_argument('--batch-size', type=int, default=512, help='input batch size for training (default: 512)')
	parser.add_argument('--optimizer', type=str, default='sgd', help='optimizer [sgd, adam]')
	parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
	parser.add_argument('--model-saving-path',  type=str, default = ".",help='path to directory where you want to save trained models (default = .)')  
	return parser

def args_parser_iterprune():
	"""
	returns argument parser object used for iterative pruning
	"""
	parser = argparse.ArgumentParser()
	parser.add_argument('--architecture', type=str, metavar='arch',required=True, help='neural network architecture [vgg19, resnet50]') 
	parser.add_argument('--target-dataset',type=str, required=True, help='dataset [cifar10, cifar100, svhn, fashionmnist]')
	parser.add_argument('--batch-size', type=int, default=512,help='input batch size for training (default: 512)')
	parser.add_argument('--source-dataset',type=str, required=True, help='dataset [cifar10, cifar100, svhn, fashionmnist]')
	parser.add_argument('--optimizer', type=str, default='sgd', help='optimizer [sgd, adam]')
	parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
	parser.add_argument('--init-path',type=str, required=True, help='path to winning initialization model (default = .)')  
	parser.add_argument('--model-saving-path', type=str, default = ".", help='path to directory where you want to save trained models (default = .)')
	parser.add_argument('--random', type=str,  default = "false", help='to train random ticket (default = false)')
	return parser


def args_parser_test():
	"""
	returns argument parser object used while testing a model
	"""
	parser = argparse.ArgumentParser()
	parser.add_argument('--architecture', type=str, metavar='arch', required=True, help='neural network architecture [vgg19, resnet50]') 
	parser.add_argument('--dataset',type=str, required=True, help='dataset [cifar10, cifar100, svhn, fashionmnist]')
	parser.add_argument('--batch-size', type=int, default=512, help='input batch size for training (default: 512)')
	parser.add_argument('--model-path',type=str, required=True, help='path to the model for finding test accuracy')
	return parser