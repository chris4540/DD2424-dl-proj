"""
Train the VGG-16 for progressive block distillation

Notes:
For running on colab, use the following code
from google.colab import drive
drive.mount('/content/drive/')  # mount google drive
cd /content/drive/My\ Drive/StudyInKTH/DD2424-DL/pytorch-cifar
"""
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

if __name__ == "__main__":
    # =============================================================
    # Settings
    role = "teacher"
    chk_pt_file = './vgg_16_{}_chkpt.tar'.format(role)
    checkpoint = None
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    resume = False
    is_eval = False
    # batch size
    batch_size = 100
    # number of epochs
    epochs = 300
    # L2 regularization weight / L2 penalty
    l2_reg_weight = 5e-4
    lr = 0.05
    # ================================================
    if resume or is_eval:
        # Load checkpoint.
        print('Loading check point file')
        checkpoint = torch.load(chk_pt_file)
    # ==========================================================================
    # cpu or gpu
    print("================================")
    print("Going to use deive : ", device)
    print("Model role: ", role)
    print("================================")
    #
    if device == 'cuda':
        cudnn.benchmark = True
    #
    if checkpoint:
        start_epoch = checkpoint['epoch']
    else:
        start_epoch = 0
    # define the img trasformation
    ###########################################################################
    ## obtain the dataset
    if not is_eval:
        trainloader, validloader = get_train_valid_cifar10_dataloader('../../data', batch_size)
    else:
        testloader = get_test_cifar10_dataloader('../../data', batch_size)
    ###########################################################################
    if role == "teacher":
        net = Vgg("VGG16")
    else:
        net = VggStudent("VGG16")

    if checkpoint:
        net.load_state_dict(checkpoint['state_dict'])

    net = net.to(device)

    if device == 'cuda':
        net.half()  # use half precision

    optimizer = optim.SGD(net.parameters(),
        lr=lr, momentum=0.9, weight_decay=l2_reg_weight)
    # scheduler = optim.lr_scheduler.CyclicLR(optimizer, 1e-5, 1e-2, mode='triangular2')
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
    best_score = -np.inf
    if is_eval:
        print("Evaluating the model with the test set")
        score = evalation(testloader, net, device)
        print("Test score: ", score)
        import sys
        sys.exit(0)

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
                'validation_score': valid_score
            }
            torch.save(saving_dict, chk_pt_file)
