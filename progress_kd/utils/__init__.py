"""
Common routine to train and evaluate the models
"""
import torch
import time
import torchvision.transforms as transforms

def get_img_tranformation():
    ret = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.4914, 0.4822, 0.4465),
                             std=(0.2023, 0.1994, 0.2010))])
    return ret

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

    n_batch = len(train_loader)
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        ts = time.time()
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
        # ========================================================
        if batch_idx % (n_batch / 10) == 0:
            et = time.time() - ts
            print("[{}/{}] Batch time used: {:.3f}".format(
                    batch_idx, n_batch, et))

    # print statistics
    train_loss = train_loss / len(train_loader)
    acc = 100 * correct / total
    used_time = time.time() - start_time
    print('Time used: %d \t Loss: %.3f | Acc: %.3f%% (%d/%d)' %
        (used_time, train_loss, acc, correct, total))