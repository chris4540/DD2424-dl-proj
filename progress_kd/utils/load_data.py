import torch
import torchvision
from utils import get_img_tranformation

def get_train_valid_cifar10_dataloader(data_dir, batch_size=100):
    trans = get_img_tranformation()
    full_dataset = torchvision.datasets.CIFAR10(
            root=data_dir, train=True, download=False, transform=trans)
    train_size = int(0.99 * len(full_dataset))
    valid_size = len(full_dataset) - train_size
    # split the training set into training and validation
    train_set, valid_set = \
        torch.utils.data.random_split(full_dataset, [train_size, valid_size])

    trainloader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=False, num_workers=2)
    validloader = torch.utils.data.DataLoader(
        valid_set, batch_size=batch_size, shuffle=False, num_workers=2)

    return trainloader, validloader

def get_test_cifar10_dataloader(data_dir, batch_size=100):
    trans = get_img_tranformation()
    testset = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=False, transform=trans)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=2)
    return testloader