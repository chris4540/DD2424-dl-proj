import torch
from torch.utils import data
import torchvision
from utils import get_img_tranformation

def get_train_valid_cifar10_dataloader(data_dir, batch_size=100, train_portion=0.99):
    trans = get_img_tranformation()
    full_dataset = torchvision.datasets.CIFAR10(
            root=data_dir, train=True, download=False, transform=trans)
    num_train = len(full_dataset)
    train_size = int(train_portion * num_train)
    valid_size = len(full_dataset) - train_size
    # splite the dataset to train and validation set non-randomly
    idxs = list(range(num_train))
    train_idxs = idxs[valid_size:]
    valid_idxs = idxs[:valid_size]
    #
    train_set = data.Subset(full_dataset, train_idxs)
    valid_set = data.Subset(full_dataset, valid_idxs)

    trainloader = data.DataLoader(
        train_set, batch_size=batch_size, shuffle=False, num_workers=2)
    validloader = data.DataLoader(
        valid_set, batch_size=batch_size, shuffle=False, num_workers=2)

    return trainloader, validloader

def get_test_cifar10_dataloader(data_dir, batch_size=100):
    trans = get_img_tranformation()
    testset = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=False, transform=trans)
    testloader = data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=2)
    return testloader