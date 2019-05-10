import torch
import torch.optim as optim
from models.vgg import Vgg
from models.vgg_aux import AuxiliaryVgg
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
    epochs = 6
    # ==============================================
    teacher = Vgg('VGG16', batch_norm=True)
    chkpt = torch.load("vgg16bn_teacher.tar")
    teacher.load_state_dict(chkpt['state_dict'])
    teacher.to(device)
    if device == 'cuda':
        teacher.half()

    # =========================================================================
    trainloader, validloader = get_train_valid_cifar10_dataloader('../../data', batch_size)
    for phase_idx in range(1, 6):
        print("phase: ", phase_idx)
        print("Making aux {} network...".format(phase_idx))
        student = AuxiliaryVgg(teacher, phase_idx, batch_norm=True)
        student.to(device)
        if device == 'cuda':
            student.half()
        optimizer = get_optimizer(student)
        scheduler = optim.lr_scheduler.CyclicLR(optimizer, 1e-5, 1e-2)
        best_score = -np.inf
        for epoch in range(0, epochs):
            print("Epoch:", epoch)
            # train for one epoch
            train(trainloader, student, optimizer, scheduler, device)

            # evaluate on validation set
            valid_score = evalation(validloader, student, device)
            print("Validation Score: ", valid_score)

            if valid_score > best_score:
                best_score = valid_score
                best_model_state_dict = student.state_dict()
                saving_dict = {
                    'epoch': epoch+1,
                    'state_dict': best_model_state_dict,
                    'validation_score': valid_score
                }
                torch.save(saving_dict, 'aux{}_chkpt.tar'.format(phase_idx))

        # update teacher with the best parameters
        teacher = student
        teacher.load_state_dict(best_model_state_dict)
