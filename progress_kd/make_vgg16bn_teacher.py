"""
Prepare VGG16 teacher with batch normalization
"""
import torch
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim
import torch.backends.cudnn as cudnn
from utils.load_data import get_train_valid_cifar10_dataloader
from utils import evalation
from utils import train
import numpy as np
#
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if device == 'cuda':
    cudnn.benchmark = True
#

if __name__ == "__main__":
    num_classes = 10
    nepochs = 20
    teacher_model_file = "vgg16bn_teacher.tar"
    # ===============================================================
    teacher = models.vgg16_bn(pretrained=True)
    teacher.get_loss = nn.CrossEntropyLoss()

    num_ftrs = teacher.classifier[6].in_features
    last_layer = nn.Linear(num_ftrs, num_classes)
    torch.nn.init.kaiming_normal_(last_layer.weight.data)
    last_layer.bias.data.zero_()
    teacher.classifier[6] = last_layer

    teacher.to(device)
    if device == 'cuda':
        teacher.half()  # use half precision
    # ===========================================================
    # set optimizer and learning rate scheduler
    l2_reg_weight = 5e-4
    optimizer = optim.SGD(teacher.parameters(),
        lr=0.05, momentum=0.9, weight_decay=l2_reg_weight)
    scheduler = optim.lr_scheduler.CyclicLR(optimizer, 1e-5, 1e-2)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    # ======================================================
    trainloader, validloader = get_train_valid_cifar10_dataloader('../../data', batch_size=100)
    best_score = -np.inf
    for epoch in range(0, nepochs):
        print("Epoch:", epoch)
        # train for one epoch
        train(trainloader, teacher, optimizer, scheduler, device)

        # evaluate on validation set
        valid_score = evalation(validloader, teacher, device)
        print("Validation Score: ", valid_score)
        if valid_score > best_score:
            # update the score
            best_score = valid_score

            saving_dict = {
                'epoch': epoch+1,
                'state_dict': teacher.state_dict(),
                'validation_score': valid_score
            }
            # save when the performance is better
            torch.save(saving_dict, teacher_model_file)
