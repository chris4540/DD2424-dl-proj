"""
Prepare VGG16 teacher with batch normalization
"""
import torch
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim
import torch.backends.cudnn as cudnn
from utils.load_data import get_train_valid_cifar10_dataloader
from utils.load_data import get_test_cifar10_dataloader
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
    batch_size = 100
    # ===============================================================
    teacher = models.vgg16_bn(pretrained=True)
    teacher.get_loss = nn.CrossEntropyLoss()
    teacher.avgpool = nn.Identity()
    classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, 10),
    )
    for m in classifier.modules():
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            nn.init.constant_(m.bias, 0)
    teacher.classifier = classifier

    teacher.to(device)
    if device == 'cuda':
        teacher.half()  # use half precision
    # ===========================================================
    # set optimizer
    l2_reg_weight = 5e-4
    optimizer = optim.SGD(teacher.parameters(),
        lr=0.05, momentum=0.9, weight_decay=l2_reg_weight)
    # ======================================================
    trainloader, validloader = get_train_valid_cifar10_dataloader('../../data', batch_size)
    # set learning rate scheduler
    step_size = 2*np.int(np.floor(len(trainloader)/batch_size))
    scheduler = optim.lr_scheduler.CyclicLR(optimizer, 1e-5, 1e-2, step_size_up=step_size)
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
            best_states = teacher.state_dict()
            saving_dict = {
                'epoch': epoch+1,
                'state_dict': best_states,
                'validation_score': valid_score
            }
            # save when the performance is better
            torch.save(saving_dict, teacher_model_file)
    # ======================================================================
    # Evaluation on test
    # ======================================================================
    # 1. update the teacher with the best model
    teacher.load_state_dict(best_states)

    # 2. test the best model
    testloader = get_test_cifar10_dataloader('../../data', batch_size)
    score = evalation(testloader, teacher, device)
    print("The test accurancy/score is :", score)