"""
Train a student with hinton's method
"""
import torch
import torch.nn as nn
import torch.optim as optim
from models.vgg import VggStudent
import torch.backends.cudnn as cudnn
from utils import progressbar
from utils import evalation
import time
from utils.load_data import get_train_valid_cifar10_dataloader
from utils.load_data import get_test_cifar10_dataloader
import numpy as np
from torch.nn.functional import softmax
from torch.nn.functional import log_softmax

class Config:
    temp = 0.5
    alpha = 0.1


def loss_fn_kd(outputs, labels, teacher_outputs, temp, alpha):
    """
    The loss function for knowledge distillation
    """
    softmax_loss = nn.CrossEntropyLoss()(outputs, labels)

    if alpha == 0:
        kd_loss = 0
    else:
        kd_loss = nn.KLDivLoss(reduction='batchmean')(
            log_softmax(outputs/temp, dim=1),
            softmax(teacher_outputs/temp, dim=1))

    ret = (1-alpha)*softmax_loss + alpha*(temp**2)*kd_loss
    return ret

def train_KD(train_loader, model, optimizer, scheduler, teacher_outs, device="cuda"):
    """
    Run one train epoch
    """

    # switch to train mode
    model.train()

    start_time = time.time()
    train_loss = 0
    total = 0
    correct = 0

    # n_batch = len(train_loader)
    for i, (inputs, targets) in enumerate(progressbar(train_loader, prefix="Training")):
        inputs = inputs.to(device)
        targets = targets.to(device)
        if device == 'cuda':
            inputs = inputs.half()

        # compute output
        outputs = model(inputs)
        teacher_outputs = teacher_outs[i]
        if device == 'cuda':
            teacher_outputs.to(device)
            teacher_outputs.half()

        loss = loss_fn_kd(outputs, targets, teacher_outputs, Config.temp, Config.alpha)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        outputs = outputs.float()
        loss = loss.float()

        # measure accuracy and record loss
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        # update the learning rate
        scheduler.step()

    # print statistics
    train_loss = train_loss / len(train_loader)
    acc = 100 * correct / total
    used_time = time.time() - start_time
    print('Train Time used: %d \t Loss: %.3f | Train Acc: %.3f%% (%d/%d)' %
        (used_time, train_loss, acc, correct, total))


#
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if device == 'cuda':
    cudnn.benchmark = True
#

if __name__ == "__main__":
    batch_size = 100
    epochs = 20*5
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
    for epoch in range(0, epochs):
        print("Epoch:", epoch)
        # train for one epoch
        train_KD(trainloader, net, optimizer, scheduler, logits, device)

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
            torch.save(saving_dict, 'student_kd.chkpt.tar')