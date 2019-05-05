"""
Train the VGG-16 for progressive block distillation

Notes:
For running on colab, use the following code
from google.colab import drive
drive.mount('/content/drive/')  # mount google drive
cd /content/drive/My\ Drive/StudyInKTH/DD2424-DL/pytorch-cifar
"""

from models.vgg import VGG
import torch
import torchvision
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.nn as nn
import time
import os
import numpy as np
import argparse

def save_checkpoint(state, filename):
    """
    Save the training model
    """
    torch.save(state, filename)

def evalation(data_loader, model, criterion):
    """
    Run evaluation

    Return:
        The accurancy
    """
    # switch to evaluate mode
    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for inputs, targets in data_loader:
            # load them to GPU
            inputs = inputs.cuda()
            targets = targets.cuda()
            inputs = inputs.half()

            # predict
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    # calculate the correct classfied
    score = correct / total
    return score

def train(train_loader, model, criterion, optimizer, scheduler):
    """
    Run one train epoch
    """

    # switch to train mode
    model.train()

    start_time = time.time()
    train_loss = 0
    total = 0
    correct = 0

    for inputs, targets in train_loader:
        inputs = inputs.cuda()
        targets = targets.cuda()
        inputs = inputs.half()

        # compute output
        output = model(inputs)
        loss = criterion(output, targets)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        output = output.float()
        loss = loss.float()

        # measure accuracy and record loss
        train_loss += loss.item()
        _, predicted = output.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        # update the learning rate
        scheduler.step()

    # print statistics
    train_loss = train_loss / len(train_loader)
    acc = 100 * correct / total
    used_time = time.time() - start_time
    print('Time used: %d \t Loss: %.3f | Acc: %.3f%% (%d/%d)' %
        (used_time, train_loss, acc, correct, total))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
    args = parser.parse_args()
    #
    chk_pt_file = os.path.join(".", 'vgg_16_checkpoint.tar')
    checkpoint = None
    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        checkpoint = torch.load(chk_pt_file)
    # ==========================================================================
    # cpu or gpu
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("================================")
    print("Going to use deive : ", device)
    print("================================")

    #
    if device == 'cuda':
        cudnn.benchmark = True

    # batch size
    batch_size = 100
    #
    epochs = 20

    #
    if checkpoint:
        start_epoch = checkpoint['epoch']
    else:
        start_epoch = 0
    # define the img trasformation
    img_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.4914, 0.4822, 0.4465),
                             std=(1.0, 1.0, 1.0))
        ])
    ###########################################################################
    ## obtain the dataset
    full_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=False, transform=img_transform)
    train_size = int(0.99 * len(full_dataset))
    valid_size = len(full_dataset) - train_size
    # split the training set into training and validation
    train_set, valid_set = \
        torch.utils.data.random_split(full_dataset, [train_size, valid_size])

    trainloader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=False, num_workers=2)
    validloader = torch.utils.data.DataLoader(
        valid_set, batch_size=batch_size, shuffle=False, num_workers=2)
    # ======================================================
    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=False, transform=img_transform)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=2)
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse',
               'ship', 'truck')
    ###########################################################################
    # initialize the model
    net = VGG('VGG16')
    if checkpoint:
        net.load_state_dict(checkpoint['state_dict'])

    net = net.to(device)
    net.half()  # use half precision

    criterion = nn.CrossEntropyLoss()
    criterion.half()
    optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9)
    scheduler = optim.lr_scheduler.CyclicLR(optimizer, 1e-5, 1e-2)
    best_score = -np.inf
    for epoch in range(start_epoch, epochs):
        print("Epoch:", epoch)
        # train for one epoch
        train(trainloader, net, criterion, optimizer, scheduler)

        # evaluate on validation set
        valid_score = evalation(validloader, net, criterion)
        print("Validation Score: ", valid_score)

        if valid_score > best_score:
            best_score = max(valid_score, best_score)
            saving_dict = {
                'epoch': epoch+1,
                'state_dict': net.state_dict(),
                'validation_score': valid_score
            }
            save_checkpoint(saving_dict, chk_pt_file)
