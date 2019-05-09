import torch
from models.vgg import Vgg
from models.vgg import VggStudent

def get_sum_params(model):
    ret = 0
    for p in model.parameters():
        ret += p.numel()
    return ret

if __name__ == "__main__":
    teacher = Vgg('VGG16')
    student = VggStudent('VGG16')
    x = torch.randn(2,3,32,32)
    y1 = teacher(x)
    y2 = student(x)
    assert y1.shape == y2.shape
    print(y1.shape)
    # print(teacher)
    # print(student)

    # # batch norm
    # t_bn = Vgg('VGG16', batch_norm=True)
    # s_bn = VggStudent('VGG16', batch_norm=True)
    # y1_bn = t_bn(x)
    # y2_bn = s_bn(x)
    # assert y1_bn.shape == y2_bn.shape
    # # print(t_bn)
    # # print(s_bn)
    print("# of params in teacher submodel:", get_sum_params(teacher))
    print("# of params in student submodel:", get_sum_params(student))
