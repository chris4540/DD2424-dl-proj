import torch
import torch.nn as nn
from models.vgg import VGG
from models.vgg import VGGStudent


if __name__ == "__main__":
    teacher = VGG('VGG16')
    student = VGGStudent('VGG16')
    block_st = 0
    block_end = -1
    for idx, f in enumerate(student.features):

        if isinstance(f, nn.MaxPool2d):
            block_end = idx
            print(block_st, block_end)
            print("==================================")
            print(student.features[block_st:block_end])
            print("==================================")
            block_st = idx+1

