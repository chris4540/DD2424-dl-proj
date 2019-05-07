import torch
from models.vgg import AuxiliaryVgg
from models.vgg import Vgg

def get_sum_params(model):
    ret = 0
    for p in model.parameters():
        ret += p.numel()
    return ret

if __name__ == "__main__":
    teacher = Vgg('VGG16')
    net = AuxiliaryVgg(teacher, 1)
    net.drop_teacher_subnet_blk()
    print("# of params = ", get_sum_params(net))
    for k in range(2, 6):
        student = AuxiliaryVgg(net, k)
        student.drop_teacher_subnet_blk()
        print("# of params = ", get_sum_params(student))
        net = student


