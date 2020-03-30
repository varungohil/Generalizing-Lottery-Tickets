from utils import *
from dataloader import *
from model import *
from parser import *
import torch
import random
import torchvision
import torch.optim as optim
import pickle

def train(model, dataloader_train, dataloader_test, dataset, architecture, optimizer_type, freeze_mask, freeze_conv, random_weights, device, models_dir, save_name):
    """
    Function to train the network 

    Arguments
    ---------
    model : the PyTorch neural network model to be trained
    dataloader : PyTorch dataloader for loading the dataset
    architecture : The neural network architecture (VGG19 or ResNet50)
    optimizer_type : The optimizer to use for training (SGD / Adam)
    device : Device(GPU/CPU) on which to perform computation
    model_path: Path to directory where trained model/checkpoints will be saved

    Returns
    -------
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
    if optimizer_type == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0001)
    elif optimizer_type == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=0.0003, weight_decay=0.0001)
    else:
        raise ValueError(optimizer_type + " optimizer not supported")

    if architecture == "vgg19":
        model.apply(initialize_xavier_normal)

    model.to(device)

    if freeze_conv:
        print('Training freezing convolutional layers')
        freeze_conv_weights(model)

    if freeze_mask:
        print('Training freezing mask of zeros')
        zeros_masks = get_zeros_masks(model)

    if random_weights:
        model = load_model(args.architecture, num_classes)
        model.to(device)
        apply_zeros_mask(model, zeros_masks, target='params')

    print(f"Started Training...", flush=True)
    for epoch in range(1, num_epochs+1):
        if epoch in lr_anneal_epochs:
            optimizer.param_groups[0]['lr'] /= 10

        correct = 0
        total = 0
        total_loss = 0
        for batch_num, data in enumerate(dataloader_train, 0):
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            if freeze_mask:
                apply_zeros_mask(model, zeros_masks, target='grad')

            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_accuracy = correct / total * 100
        train_loss = total_loss / total
        print(f"[TRAINING] Epoch {epoch} : Accuracy {train_accuracy} : Loss = {train_loss}", flush=True)

        if epoch % 3 == 0:
            test_loss, test_accuracy = test(model, dataloader_test)
            print(f'Epoch {epoch} : Test Accuracy {test_accuracy} : Test Loss {test_loss}')

        if epoch == num_epochs:
            try:
                torch.save({'epoch': epoch,'model_state_dict': model.state_dict(),'optimizer_state_dict': optimizer.state_dict()}, models_dir + f"/{save_name}")
            except FileNotFoundError:
                print(models_dir + " path not found", flush=True)
        
    test_loss, test_accuracy = test(model, dataloader_test)
    print(f'Test Loss {test_loss} : Test Accuracy {test_accuracy}')
    print("Finished Training!")
    return test_accuracy

def test(model, dataloader):
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

    zero_total = 0
    zeros = 0
    for name, params in model.named_parameters():
        if "weight" in name:
            weight_copy = params.data.abs().clone()
            zeros += weight_copy.numel() - weight_copy.nonzero().size(0)
            zero_total += weight_copy.numel()
    print(f"Fraction of weights pruned = {zeros}/{zero_total} = {zeros/zero_total}")

    criterion = nn.CrossEntropyLoss().cuda()

    correct = 0
    total = 0
    total_loss = 0
    with torch.no_grad():
        for data in dataloader:
            inputs, labels = data[0].to(device), data[1].to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    test_loss = round(total_loss / total, 4)
    test_accuracy = round(correct / total * 100, 3)
    return test_loss, test_accuracy

if __name__ == '__main__':
    #Parsers the command line arguments
    parser = args_parser_train()
    args = parser.parse_args()

    #Sets random seed
    random.seed(args.seed)

    #Uses GPU is available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'Using {device} device.', flush=True)

    #Loads dataset
    dataloader_train = load_dataset(args.dataset, args.batch_size, True)
    dataloader_test = load_dataset(args.dataset, args.batch_size, False)

    #Checks number of classes to aa appropriate linear layer at end of model
    if args.dataset in ['cifar10', 'fashionmnist', 'svhn']:
        num_classes = 10
    elif args.dataset in ['cifar100']:
        num_classes = 100
    else:
        raise ValueError(args.dataset + " dataset not supported")

    #Loads model
    model = load_model(args.architecture, num_classes)

    # Pre-training
    save_name = '90'
    if args.init_path:
        source_dataset = args.init_path.split('/')[-2].split('_')[1]
        save_name = args.init_path.split('/')[-1]
        if source_dataset in ['cifar10', 'fashionmnist', 'svhn']:
            source_num_classes = 10
        elif source_dataset in ['cifar100']:
            source_num_classes = 100
        is_equal_classes = num_classes == source_num_classes
        load_weights(model, args.init_path, is_equal_classes)

    # Print args
    print(args)

    calculate_sparsity(model)
    test_accuracy = train(model, dataloader_train, dataloader_test, args.dataset, args.architecture, args.optimizer, args.freeze_mask, args.freeze_conv, args.random_weights, device, args.model_saving_path, save_name)
    calculate_sparsity(model)







