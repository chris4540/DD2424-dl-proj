import torch
from models.vgg import AuxiliaryVgg

if __name__ == "__main__":
    net = AuxiliaryVgg('VGG16', 1)
    x = torch.randn(2,3,32,32)
    y = net(x)

    net = AuxiliaryVgg('VGG16', 5)
    for name, p in net.named_parameters():
        print(name, p.requires_grad)