"""
Common routine to train and evaluate the models
"""
import torch
import torch.nn as nn
import time
import torchvision.transforms as transforms
import sys

def progressbar(it, prefix="", size=40, file=sys.stdout):
    """
    Progress bar showing
    Args:
        TODO
    Return:
        the yeild from the input iterator/generator

    See also:
        https://stackoverflow.com/a/34482761
    """
    count = len(it)
    def show(j):
        x = int(size*j/count)
        file.write("%s[%s%s] %i/%i\r" % (prefix, "#"*x, "."*(size-x), j, count))
        file.flush()
    show(0)
    for i, item in enumerate(it):
        yield item
        show(i+1)
    file.write("\n")
    file.flush()

def get_img_tranformation():
    ret = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize(mean=(0.4914, 0.4822, 0.4465),
        #                      std=(0.2023, 0.1994, 0.2010))
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
        ])
    return ret

def evalation(data_loader, model, device='cuda'):
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
        for inputs, targets in progressbar(data_loader, prefix="Evaluating"):
            # load them to GPU
            inputs = inputs.to(device)
            targets = targets.to(device)
            if device == 'cuda':
                inputs = inputs.half()

            # predict
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    # calculate the correct classfied
    score = correct / total
    return score

def train(train_loader, model, optimizer, scheduler, device="cuda"):
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
    for inputs, targets in progressbar(train_loader, prefix="Training"):
        inputs = inputs.to(device)
        targets = targets.to(device)
        if device == 'cuda':
            inputs = inputs.half()

        # compute output
        output = model(inputs)
        loss = model.get_loss(output, targets)

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
    print('Train Time used: %d \t Loss: %.3f | Train Acc: %.3f%% (%d/%d)' %
        (used_time, train_loss, acc, correct, total))
