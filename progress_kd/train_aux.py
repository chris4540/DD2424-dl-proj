import torch
import torch.optim as optim
from models.vgg import Vgg
from models.vgg import AuxiliaryVgg
from utils import train
from utils import evalation
from utils.load_data import get_train_valid_cifar10_dataloader
from utils.load_data import get_test_cifar10_dataloader
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_optimizer(model):
    params_to_update = []
    print("Params to be updated")
    for name, p in model.named_parameters():
        if p.requires_grad:
            params_to_update.append(p)
            print(name)
    optimizer = optim.SGD(params_to_update, lr=0.1, momentum=0.9)
    return optimizer

if __name__ == "__main__":

    batch_size = 100
    # ==============================================
    teacher = Vgg('VGG16')
    chkpt = torch.load("vgg_16_teacher_chkpt.tar")
    teacher.load_state_dict(chkpt['state_dict'])
    teacher.to(device)
    if device == 'cuda':
        teacher.half()

    # testloader = get_test_cifar10_dataloader('../../data', batch_size=100)

    # score = evalation(testloader, teacher, device=device)
    # print("Teacher score: ", score)

    # train first aux
    aux1 = AuxiliaryVgg(teacher, 1)
    aux1.to(device)
    if device == 'cuda':
        aux1.half()

    # =========================================================================
    trainloader, validloader = get_train_valid_cifar10_dataloader('../../data', batch_size)
    optimizer = get_optimizer(aux1)
    scheduler = optim.lr_scheduler.CyclicLR(optimizer, 1e-5, 1e-2)
    best_score = -np.inf
    for epoch in range(0, 20):
        print("Epoch:", epoch)
        # train for one epoch
        train(trainloader, aux1, optimizer, scheduler, device)

        # evaluate on validation set
        valid_score = evalation(validloader, aux1, device)
        print("Validation Score: ", valid_score)

        if valid_score > best_score:
            best_score = max(valid_score, best_score)
            saving_dict = {
                'epoch': epoch+1,
                'state_dict': aux1.state_dict(),
                'validation_score': valid_score
            }
            torch.save(saving_dict, 'aux1_chkpt.tar')

