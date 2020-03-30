import torch
import numpy as np
import torch.nn as nn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
iterations_testing_epochs = [0, 1, 2, 3, 4, 5, 6, 9, 12, 15, 18, 21, 24, 27, 30]

def load_weights(model, init_path, is_equal_classes):
    cpt = torch.load(init_path)
    if is_equal_classes:
        print('Loading full weights')
        model.load_state_dict(cpt['model_state_dict'])
    else:
        print('Loading convolutional weights only')
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
    print(f'Loaded weights from {init_path}', flush=True)

def freeze_conv_weights(model):
    layer_index = 0
    for name, params in model.named_parameters():
        if "classifier" not in name and "fc" not in name and 'linear' not in name:
            params.requires_grad = False
        layer_index += 1

def get_zeros_masks(model):
    zeros_masks = []
    for name, params in model.named_parameters():
        if "weight" in name:
            weight_copy = params.data.abs().clone()
            mask = (weight_copy != 0).float().to(device)
            zeros_masks.append(mask)
    return zeros_masks

def apply_zeros_mask(model, masks, target='grad'):
    layer_index = 0
    for name, params in model.named_parameters():
        if 'weight' in name:
            if target == 'grad' and params.requires_grad: 
                params.grad *= masks[layer_index]
            if target == 'params':
                params.data *= masks[layer_index]
            layer_index += 1


def calculate_sparsity(model):
    zero_total = 0
    zeros = 0
    for name, params in model.named_parameters():
        if "weight" in name:
            weight_copy = params.data.abs().clone()
            zeros += weight_copy.numel() - weight_copy.nonzero().size(0)
            zero_total += weight_copy.numel()
            print(f"{name} {(weight_copy.numel() - weight_copy.nonzero().size(0)) / weight_copy.numel()} with shape {params.shape}")
    print(f"Fraction of weights pruned = {zeros}/{zero_total} = {zeros/zero_total}", flush=True)

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
