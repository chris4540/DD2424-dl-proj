"""
Train a student from scatch
"""
import torch
import torch.nn as nn
import torch.optim as optim
from models.vgg import VggStudent
import torch.backends.cudnn as cudnn
from utils import progressbar
from utils import evalation
from utils import train
from utils.load_data import get_train_valid_cifar10_dataloader
from utils.load_data import get_test_cifar10_dataloader
import numpy as np


#
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if device == 'cuda':
    cudnn.benchmark = True
#

if __name__ == "__main__":
    batch_size = 100
    epochs = 50
    net = VggStudent("VGG16", batch_norm=True)
    net.to(device)
    if device == 'cuda':
        net.half()

    # load teacher logits
    logit_data = torch.load("teacher_logits_100.tar")
    assert logit_data['batch_size'] == batch_size
    logits = logit_data['logits']

    # ===========================================================================
    trainloader, validloader = get_train_valid_cifar10_dataloader('../../data', batch_size)
    testloader = get_test_cifar10_dataloader('../../data', batch_size)
    step_size = 2*np.int(np.floor(len(trainloader)/batch_size))
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CyclicLR(optimizer, 1e-5, 1e-2, step_size_up=step_size)
    best_score = -np.inf
    for epoch in range(0, epochs):
        print("Epoch:", epoch)
        # train for one epoch
        train(trainloader, net, optimizer, scheduler, device)

        # evaluate on validation set
        valid_score = evalation(validloader, net, device)
        print("Validation Score: ", valid_score)

        if valid_score > best_score:
            best_score = valid_score
            best_model_state_dict = net.state_dict()
            saving_dict = {
                'epoch': epoch+1,
                'state_dict': best_model_state_dict,
                'validation_score': valid_score
            }
            torch.save(saving_dict, 'student.scratch.tar')

    # test
    # update teacher with the best parameters
    net.load_state_dict(best_model_state_dict)
    teacher = net
    score = evalation(testloader, net, device)
    print("The test accurancy/score is :", score)