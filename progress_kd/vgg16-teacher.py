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
    chk_pt_file = './vgg_16_teacher_chkpt.tar'
    checkpoint = None
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    resume = False
    is_eval = True
    # batch size
    batch_size = 100
    #
    epochs = 20
    # ================================================
    if resume or is_eval:
        # Load checkpoint.
        print('Loading check point file')
        checkpoint = torch.load(chk_pt_file)
    # ==========================================================================
    # cpu or gpu
    print("================================")
    print("Going to use deive : ", device)
    print("================================")
    # img_transform = get_img_tranformation()
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
    if is_eval:
        score = evalation(testloader, net, criterion)
        print("Test score: ", score)
        import os
        os.exit(0)

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
            torch.save(saving_dict, chk_pt_file)
