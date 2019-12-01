import argparse
import os
import sys
import math
import torch
import torchvision
from torch.utils.data import Dataset
import torch.optim as optim
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.hub import load_state_dict_from_url

def args_parser_train():
	"""
	returns argument parser object used while training a model
	"""
    parser = argparse.ArgumentParser()

    parser.add_argument('--architecture', required=True, type=str,
                        help='neural network architecture [vgg19, resnet50]') 

    parser.add_argument('--dataset',type=str, required=True,
                        help='dataset [cifar10, cifar100, svhn, fashionmnist]')

    parser.add_argument('--batch-size', type=int, default=512,
                        help='input batch size for training (default: 512)')

    parser.add_argument('--optimizer', type=str, default='sgd',
                        help='optimizer [sgd, adam]')

    parser.add_argument('--seed', type=int, default=1,
                        help='random seed (default: 1)')

    parser.add_argument('--model-saving-path',  type=str, default = ".",
                        help='path to directory where you want to save trained models (default = .)')  
    return parser

def args_parser_iterprune():
	"""
	returns argument parser object used for iterative pruning
	"""
    parser = argparse.ArgumentParser()

    parser.add_argument('--architecture', type=str, metavar='arch',required=True,
                        help='neural network architecture [vgg19, resnet50]') 

    parser.add_argument('--target-dataset',type=str, required=True,
                        help='dataset [cifar10, cifar100, svhn, fashionmnist]')
    parser.add_argument('--batch-size', type=int, default=512,
                        help='input batch size for training (default: 512)')

    parser.add_argument('--source-dataset',type=str, required=True,
                        help='dataset [cifar10, cifar100, svhn, fashionmnist]')

    parser.add_argument('--optimizer', type=str, default='sgd',
                        help='optimizer [sgd, adam]')

    parser.add_argument('--seed', type=int, default=1,
                        help='random seed (default: 1)')

    parser.add_argument('--init-path',type=str, required=True,
                        help='path to winning initialization model (default = .)')  
  
    parser.add_argument('--model-saving-path', type=str, default = ".",
                        help='path to directory where you want to save trained models (default = .)')  

    parser.add_argument('--random', type=str,  default = "false",
                        help='to train random ticket (default = false)')
    return parser


def args_parser_test():
	"""
	returns argument parser object used while testing a model
	"""
    parser = argparse.ArgumentParser()

    parser.add_argument('--architecture', type=str, metavar='arch', required=True,
                        help='neural network architecture [vgg19, resnet50]') 

    parser.add_argument('--dataset',type=str, required=True,
                        help='dataset [cifar10, cifar100, svhn, fashionmnist]')

    parser.add_argument('--batch-size', type=int, default=512,
                        help='input batch size for training (default: 512)')

    parser.add_argument('--model_path',type=str, required=True,
                        help='path to the model for finding test accuracy')  
    return parser

def load_dataset(dataset, batch_size = 512, is_train_split=True):
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
			data_set = torchvision.datasets.CIFAR10(root='../datasets', train=is_train_split, download=True, transform=transforms.Compose([transform_augment, transform]))
			data_loader = torch.utils.data.DataLoader(data_set, batch_size=batch_size, shuffle=True)
		elif dataset == 'cifar100':
			transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
			transform_augment = transforms.Compose([transforms.RandomHorizontalFlip(), transforms. RandomCrop(32, padding=4)])
			data_set = torchvision.datasets.CIFAR100(root='../datasets', train=is_train_split, download=True, transform=transforms.Compose([transform_augment, transform]))
			data_loader = torch.utils.data.DataLoader(data_set, batch_size=batch_size, shuffle=True)
		elif dataset == 'svhn':
			if not is_train_split:
				svhn_split = 'test'
			else:
				svhn_split = 'train'
			transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
			transform_augment = transforms.Compose([transforms.RandomHorizontalFlip(), transforms. RandomCrop(32, padding=4)])
			data_set = torchvision.datasets.SVHN(root='../datasets', split=svhn_split , download=True, transform=transforms.Compose([transform_augment, transform]))
			data_loader = torch.utils.data.DataLoader(data_set, batch_size=batch_size, shuffle=True)
		elif dataset == 'fashionmnist':
			transform = transforms.Compose([transforms.Grayscale(3),transforms.ToTensor(),transforms.Normalize((0.5,), (0.5,))])
			transform_augment = transforms.Compose([transforms.RandomHorizontalFlip(), transforms. RandomCrop(32, padding=4)])
			data_set = torchvision.datasets.FashionMNIST(root='../datasets', train=is_train_split, download=True, transform=transforms.Compose([transform_augment, transform]))
			data_loader = torch.utils.data.DataLoader(data_set, batch_size=batch_size, shuffle=True, num_workers=2)
		elif dataset == 'cifar10a' or dataset == 'cifar10b':
			transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
			transform_augment = transforms.Compose([transforms.RandomHorizontalFlip(), transforms. RandomCrop(32, padding=4)])
			cifarset = torchvision.datasets.CIFAR10(root='../datasets', train=is_train_split, download=True, transform=transforms.Compose([transform_augment, transform]))
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
			data_set = torchvision.datasets.CIFAR10(root='../datasets', train=is_train_split, download=True, transform=transform)
			data_loader = torch.utils.data.DataLoader(data_set, batch_size=batch_size, shuffle=True)
		elif dataset == 'cifar100':
			transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
			# transform_augment = transforms.Compose([transforms.RandomHorizontalFlip(), transforms. RandomCrop(32, padding=4)])
			data_set = torchvision.datasets.CIFAR100(root='../datasets', train=is_train_split, download=True, transform=transform)
			data_loader = torch.utils.data.DataLoader(data_set, batch_size=batch_size, shuffle=True)
		elif dataset == 'svhn':
			if not is_train_split:
				svhn_split = 'test'
			else:
				svhn_split = 'train'
			transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
			# transform_augment = transforms.Compose([transforms.RandomHorizontalFlip(), transforms. RandomCrop(32, padding=4)])
			data_set = torchvision.datasets.SVHN(root='../datasets', split=svhn_split , download=True, transform=transform)
			data_loader = torch.utils.data.DataLoader(data_set, batch_size=batch_size, shuffle=True)
		elif dataset == 'fashionmnist':
			transform = transforms.Compose([transforms.Grayscale(3),transforms.ToTensor(),transforms.Normalize((0.5,), (0.5,))])
			# transform_augment = transforms.Compose([transforms.RandomHorizontalFlip(), transforms. RandomCrop(32, padding=4)])
			data_set = torchvision.datasets.FashionMNIST(root='../datasets', train=is_train_split, download=True, transform=transform)
			data_loader = torch.utils.data.DataLoader(data_set, batch_size=batch_size, shuffle=True, num_workers=2)
		elif dataset == 'cifar10a' or dataset == 'cifar10b':
			transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
			transform_augment = transforms.Compose([transforms.RandomHorizontalFlip(), transforms. RandomCrop(32, padding=4)])
			cifarset = torchvision.datasets.CIFAR10(root='../datasets', train=is_train_split, download=True, transform=transforms.Compose([transform_augment, transform]))
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


def load_model(architecture, num_classes):
	"""
	Loads the neural network model. 
	The definitions of architectures are taken from PyTorch source code.  

	Arguments
	---------
	architecture : The neural network architecture
	num_classes  : The number of classes in dataset on which the model will be trained

	Returns
	------
	The PyTorch neural network model object
	"""
	if architecture == "vgg19":
		__all__ = ['VGG', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
		           'vgg19', 'vgg19_bn']

		class VGG(nn.Module):
		    '''
		        VGG Model
		    '''

		    def __init__(self, features):

		        super(VGG, self).__init__()
		        self.features = features
		        self.classifier = nn.Sequential(
		            nn.Linear(512, num_classes)
		        )

		        # Initialize weights
		        for m in self.modules():
		            if isinstance(m, nn.Conv2d):
		                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
		                m.weight.data.normal_(0, math.sqrt(2. / n))
		                m.bias.data.zero_()

		    def forward(self, x):
		        x = self.features(x)
		        x = x.view(x.size(0), -1)
		        x = self.classifier(x)
		        return x

		def make_layers(cfg, batch_norm=False):

		    layers = []
		    in_channels = 3
		    for v in cfg:
		        if v == 'M':
		            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
		        elif v == 'G':
		            layers+= [nn.AdaptiveAvgPool2d(1)]
		        else:
		            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
		            if batch_norm:
		                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
		            else:
		                layers += [conv2d, nn.ReLU(inplace=True)]
		            in_channels = v
		        


		    return nn.Sequential(*layers)

		cfg = {
		    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
		    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
		    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
		    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M',
		          512, 512, 512, 512, 'G'],
		}

		def vgg11():
		    """VGG 11-layer model (configuration "A")"""
		    return VGG(make_layers(cfg['A']))


		def vgg11_bn():
		    """VGG 11-layer model (configuration "A") with batch normalization"""
		    return VGG(make_layers(cfg['A'], batch_norm=True))


		def vgg13():
		    """VGG 13-layer model (configuration "B")"""
		    return VGG(make_layers(cfg['B']))


		def vgg13_bn():
		    """VGG 13-layer model (configuration "B") with batch normalization"""
		    return VGG(make_layers(cfg['B'], batch_norm=True))


		def vgg16():
		    """VGG 16-layer model (configuration "D")"""
		    return VGG(make_layers(cfg['D']))


		def vgg16_bn():
		    """VGG 16-layer model (configuration "D") with batch normalization"""
		    return VGG(make_layers(cfg['D'], batch_norm=True))


		def vgg19():
		    """VGG 19-layer model (configuration "E")"""
		    return VGG(make_layers(cfg['E']))


		def vgg19_bn():
		    """VGG 19-layer model (configuration 'E') with batch normalization"""
		    return VGG(make_layers(cfg['E'], batch_norm=True))

		return vgg19_bn()

	elif architecture == "resnet50":
		__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
		          'resnet152', 'resnext50_32x4d', 'resnext101_32x8d',
		           'wide_resnet50_2', 'wide_resnet101_2']


		model_urls = {
		    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
		    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
		    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
		    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
		    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
		    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
		    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
		    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
		    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
		}


		def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
		    """3x3 convolution with padding"""
		    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
		                     padding=dilation, groups=groups, bias=False, dilation=dilation)


		def conv1x1(in_planes, out_planes, stride=1):
		    """1x1 convolution"""
		    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


		class BasicBlock(nn.Module):
		    expansion = 1
		    __constants__ = ['downsample']

		    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
		                 base_width=64, dilation=1, norm_layer=None):
		        super(BasicBlock, self).__init__()
		        if norm_layer is None:
		            norm_layer = nn.BatchNorm2d
		        if groups != 1 or base_width != 64:
		            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
		        if dilation > 1:
		            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
		        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
		        self.conv1 = conv3x3(inplanes, planes, stride)
		        self.bn1 = norm_layer(planes)
		        self.relu = nn.ReLU(inplace=True)
		        self.conv2 = conv3x3(planes, planes)
		        self.bn2 = norm_layer(planes)
		        self.downsample = downsample
		        self.stride = stride

		    def forward(self, x):
		        identity = x

		        out = self.conv1(x)
		        out = self.bn1(out)
		        out = self.relu(out)

		        out = self.conv2(out)
		        out = self.bn2(out)

		        if self.downsample is not None:
		            identity = self.downsample(x)

		        out += identity
		        out = self.relu(out)

		        return out


		class Bottleneck(nn.Module):
		    expansion = 4
		    __constants__ = ['downsample']

		    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
		                 base_width=64, dilation=1, norm_layer=None):
		        super(Bottleneck, self).__init__()
		        if norm_layer is None:
		            norm_layer = nn.BatchNorm2d
		        width = int(planes * (base_width / 64.)) * groups
		        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
		        self.conv1 = conv1x1(inplanes, width)
		        self.bn1 = norm_layer(width)
		        self.conv2 = conv3x3(width, width, stride, groups, dilation)
		        self.bn2 = norm_layer(width)
		        self.conv3 = conv1x1(width, planes * self.expansion)
		        self.bn3 = norm_layer(planes * self.expansion)
		        self.relu = nn.ReLU(inplace=True)
		        self.downsample = downsample
		        self.stride = stride

		    def forward(self, x):
		        identity = x

		        out = self.conv1(x)
		        out = self.bn1(out)
		        out = self.relu(out)

		        out = self.conv2(out)
		        out = self.bn2(out)
		        out = self.relu(out)

		        out = self.conv3(out)
		        out = self.bn3(out)

		        if self.downsample is not None:
		            identity = self.downsample(x)

		        out += identity
		        out = self.relu(out)

		        return out


		class ResNet(nn.Module):

		    def __init__(self, block, layers, num_classes=num_classes, zero_init_residual=False,
		                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
		                 norm_layer=None):
		        super(ResNet, self).__init__()
		        if norm_layer is None:
		            norm_layer = nn.BatchNorm2d
		        self._norm_layer = norm_layer

		        self.inplanes = 64
		        self.dilation = 1
		        if replace_stride_with_dilation is None:
		            # each element in the tuple indicates if we should replace
		            # the 2x2 stride with a dilated convolution instead
		            replace_stride_with_dilation = [False, False, False]
		        if len(replace_stride_with_dilation) != 3:
		            raise ValueError("replace_stride_with_dilation should be None "
		                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
		        self.groups = groups
		        self.base_width = width_per_group
		        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
		                               bias=False)
		        self.bn1 = norm_layer(self.inplanes)
		        self.relu = nn.ReLU(inplace=True)
		        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
		        self.layer1 = self._make_layer(block, 64, layers[0])
		        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
		                                       dilate=replace_stride_with_dilation[0])
		        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
		                                       dilate=replace_stride_with_dilation[1])
		        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
		                                       dilate=replace_stride_with_dilation[2])
		        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
		        self.fc = nn.Linear(512 * block.expansion, num_classes)

		        for m in self.modules():
		            if isinstance(m, nn.Conv2d):
		                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
		            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
		                nn.init.constant_(m.weight, 1)
		                nn.init.constant_(m.bias, 0)

		        # Zero-initialize the last BN in each residual branch,
		        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
		        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
		        if zero_init_residual:
		            for m in self.modules():
		                if isinstance(m, Bottleneck):
		                    nn.init.constant_(m.bn3.weight, 0)
		                elif isinstance(m, BasicBlock):
		                    nn.init.constant_(m.bn2.weight, 0)

		    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
		        norm_layer = self._norm_layer
		        downsample = None
		        previous_dilation = self.dilation
		        if dilate:
		            self.dilation *= stride
		            stride = 1
		        if stride != 1 or self.inplanes != planes * block.expansion:
		            downsample = nn.Sequential(
		                conv1x1(self.inplanes, planes * block.expansion, stride),
		                norm_layer(planes * block.expansion),
		            )

		        layers = []
		        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
		                            self.base_width, previous_dilation, norm_layer))
		        self.inplanes = planes * block.expansion
		        for _ in range(1, blocks):
		            layers.append(block(self.inplanes, planes, groups=self.groups,
		                                base_width=self.base_width, dilation=self.dilation,
		                                norm_layer=norm_layer))

		        return nn.Sequential(*layers)

		    def _forward(self, x):
		        x = self.conv1(x)
		        x = self.bn1(x)
		        x = self.relu(x)
		        x = self.maxpool(x)

		        x = self.layer1(x)
		        x = self.layer2(x)
		        x = self.layer3(x)
		        x = self.layer4(x)

		        x = self.avgpool(x)
		        x = torch.flatten(x, 1)
		        x = self.fc(x)

		        return x

		    # Allow for accessing forward method in a inherited class
		    forward = _forward


		def _resnet(arch, block, layers, pretrained, progress, **kwargs):
		    model = ResNet(block, layers, **kwargs)
		    if pretrained:
		        state_dict = load_state_dict_from_url(model_urls[arch],
		                                              progress=progress)
		        model.load_state_dict(state_dict)
		    return model


		def resnet18(pretrained=False, progress=True, **kwargs):
		    r"""ResNet-18 model from
		    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
		    Args:
		        pretrained (bool): If True, returns a model pre-trained on ImageNet
		        progress (bool): If True, displays a progress bar of the download to stderr
		    """
		    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress,
		                   **kwargs)


		def resnet34(pretrained=False, progress=True, **kwargs):
		    r"""ResNet-34 model from
		    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
		    Args:
		        pretrained (bool): If True, returns a model pre-trained on ImageNet
		        progress (bool): If True, displays a progress bar of the download to stderr
		    """
		    return _resnet('resnet34', BasicBlock, [3, 4, 6, 3], pretrained, progress,
		                   **kwargs)


		def resnet50(pretrained=False, progress=True, **kwargs):
		    r"""ResNet-50 model from
		    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
		    Args:
		        pretrained (bool): If True, returns a model pre-trained on ImageNet
		        progress (bool): If True, displays a progress bar of the download to stderr
		    """
		    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress,
		                   **kwargs)


		def resnet101(pretrained=False, progress=True, **kwargs):
		    r"""ResNet-101 model from
		    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
		    Args:
		        pretrained (bool): If True, returns a model pre-trained on ImageNet
		        progress (bool): If True, displays a progress bar of the download to stderr
		    """
		    return _resnet('resnet101', Bottleneck, [3, 4, 23, 3], pretrained, progress,
		                   **kwargs)


		def resnet152(pretrained=False, progress=True, **kwargs):
		    r"""ResNet-152 model from
		    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
		    Args:
		        pretrained (bool): If True, returns a model pre-trained on ImageNet
		        progress (bool): If True, displays a progress bar of the download to stderr
		    """
		    return _resnet('resnet152', Bottleneck, [3, 8, 36, 3], pretrained, progress,
		                   **kwargs)


		def resnext50_32x4d(pretrained=False, progress=True, **kwargs):
		    r"""ResNeXt-50 32x4d model from
		    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_
		    Args:
		        pretrained (bool): If True, returns a model pre-trained on ImageNet
		        progress (bool): If True, displays a progress bar of the download to stderr
		    """
		    kwargs['groups'] = 32
		    kwargs['width_per_group'] = 4
		    return _resnet('resnext50_32x4d', Bottleneck, [3, 4, 6, 3],
		                   pretrained, progress, **kwargs)


		def resnext101_32x8d(pretrained=False, progress=True, **kwargs):
		    r"""ResNeXt-101 32x8d model from
		    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_
		    Args:
		        pretrained (bool): If True, returns a model pre-trained on ImageNet
		        progress (bool): If True, displays a progress bar of the download to stderr
		    """
		    kwargs['groups'] = 32
		    kwargs['width_per_group'] = 8
		    return _resnet('resnext101_32x8d', Bottleneck, [3, 4, 23, 3],
		                   pretrained, progress, **kwargs)


		def wide_resnet50_2(pretrained=False, progress=True, **kwargs):
		    r"""Wide ResNet-50-2 model from
		    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_
		    The model is the same as ResNet except for the bottleneck number of channels
		    which is twice larger in every block. The number of channels in outer 1x1
		    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
		    channels, and in Wide ResNet-50-2 has 2048-1024-2048.
		    Args:
		        pretrained (bool): If True, returns a model pre-trained on ImageNet
		        progress (bool): If True, displays a progress bar of the download to stderr
		    """
		    kwargs['width_per_group'] = 64 * 2
		    return _resnet('wide_resnet50_2', Bottleneck, [3, 4, 6, 3],
		                   pretrained, progress, **kwargs)


		def wide_resnet101_2(pretrained=False, progress=True, **kwargs):
		    r"""Wide ResNet-101-2 model from
		    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_
		    The model is the same as ResNet except for the bottleneck number of channels
		    which is twice larger in every block. The number of channels in outer 1x1
		    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
		    channels, and in Wide ResNet-50-2 has 2048-1024-2048.
		    Args:
		        pretrained (bool): If True, returns a model pre-trained on ImageNet
		        progress (bool): If True, displays a progress bar of the download to stderr
		    """
		    kwargs['width_per_group'] = 64 * 2
		    return _resnet('wide_resnet101_2', Bottleneck, [3, 4, 23, 3],
		                   pretrained, progress, **kwargs)

		return resnet50(pretrained=False)
	else:
		raise ValueError(architecture + " architecture not supported.")