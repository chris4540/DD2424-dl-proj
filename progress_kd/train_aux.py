import torch
from models.vgg import Vgg
from models.vgg import AuxiliaryVgg
from utils import evalation
from utils.load_data import get_train_valid_cifar10_dataloader
from utils.load_data import get_test_cifar10_dataloader

device = 'cuda' if torch.cuda.is_available() else 'cpu'

if __name__ == "__main__":

    teacher = Vgg('VGG16')
    chkpt = torch.load("vgg_16_teacher_chkpt.tar")
    teacher.load_state_dict(chkpt['state_dict'])
    teacher.to(device)
    if device == 'cuda':
        teacher.half()

    testloader = get_test_cifar10_dataloader('../../data', batch_size=100)

    score = evalation(testloader, teacher, device=device)
    print("Teacher score: ", score)

    # train first aux
    a1 = AuxiliaryVgg(teacher, 1)
    a1.to(device)
    if device == 'cuda':
        a1.half()
    score = evalation(testloader, a1, device=device)
    print("Aux 1 score: ", score)

