from models.vgg import VggStudent
from models.vgg import Vgg
import torch
import torchvision
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.nn as nn
import time
import os
import numpy as np
from utils import evalation
from utils import train
from utils import get_img_tranformation
from utils.load_data import get_train_valid_cifar10_dataloader
from utils.load_data import get_test_cifar10_dataloader
import argparse

device = 'cuda' if torch.cuda.is_available() else 'cpu'
if device == 'cuda':
    cudnn.benchmark = True

def run_test(testloader, net, device):
    print("Evaluating the model with the test set")
    score = evalation(testloader, net, device)
    print("Test score: ", score)

if __name__ == "__main__":
    # Settings
    checkpoint = None
    # batch size
    batch_size = 100
    # L2 regularization weight / L2 penalty
    l2_reg_weight = 5e-4
    lr = 0.05
    # ========================================================================
    # arg parse
    is_resume = False
    is_eval = False
    is_batchnorm = True
    role = "student"
    epochs = 200
    # =================================================================
    if is_batchnorm:
        model_name = 'vgg16bn'
    else:
        model_name = 'vgg16'
    # ================================================
    # ==========================================================================
    # cpu or gpu
    print("================================")
    print("Going to use deive : ", device)
    print("Model role: ", role)
    print("Model name: ", model_name)
    print("================================")
    #
    if checkpoint:
        start_epoch = checkpoint['epoch']
    else:
        start_epoch = 0
    # define the img trasformation
    ###########################################################################
    ## obtain the dataset

    trainloader, validloader = get_train_valid_cifar10_dataloader('../../data', batch_size)
    testloader = get_test_cifar10_dataloader('../../data', batch_size)
    ###########################################################################
    if role == "teacher":
        net = Vgg("VGG16", batch_norm=is_batchnorm)
    else:
        net = VggStudent("VGG16", batch_norm=is_batchnorm)

    optimizer = optim.SGD(net.parameters(),
        lr=lr, momentum=0.9, weight_decay=l2_reg_weight)

    if checkpoint:
        net.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])

    net = net.to(device)

    if device == 'cuda':
        net.half()  # use half precision

    if is_eval:
        run_test(testloader, net, device)
        import sys
        sys.exit(0)

    # calculate step size
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    best_score = -np.inf

    for epoch in range(start_epoch, epochs):
        print("Epoch:", epoch)
        # train for one epoch
        train(trainloader, net, optimizer, scheduler, device)

        # evaluate on validation set
        valid_score = evalation(validloader, net, device)
        print("Validation Score: ", valid_score)

        if valid_score > best_score:
            best_score = max(valid_score, best_score)
            saving_dict = {
                'epoch': epoch+1,
                'state_dict': net.state_dict(),
                'optimizer': optimizer.state_dict(),
                'validation_score': valid_score,
            }
            chk_pt_file = './{}_{}_ct.tar.StepLR.{}'.format(model_name, role, epoch+1)
            torch.save(saving_dict, chk_pt_file)
    # ====================================================================
    run_test(testloader, net, device)