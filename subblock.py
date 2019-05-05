# -*- coding: utf-8 -*-
"""
Basic checking to ensure teacher subnet and the student subnet
produce a equal sized intermediate img
"""

import torch
import torch.nn as nn

def get_sum_params(model):
    ret = 0
    for p in model.parameters():
        ret += p.numel()
    return ret

if __name__ == "__main__":
    # The first block of the teacher net
    img = torch.randn(2,3,32,32)
    teacher_subnet = [
        nn.Conv2d(3, 64, kernel_size=3, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(inplace=True),
        nn.Conv2d(64, 64, kernel_size=3, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2)
    ]
    teacher_submodel = nn.Sequential(*teacher_subnet)
    inter_img1 = teacher_submodel(torch.randn(2,3,32,32))
    print("The size of the teacher subnet output: ", inter_img1.shape)

    # The first block of the teacher net
    # student
    student_block = [
        nn.Conv2d(3, 32, kernel_size=3, padding=1),
        nn.BatchNorm2d(32),
        nn.ReLU(inplace=True),
        nn.Conv2d(32, 32, kernel_size=3, padding=1),
        nn.BatchNorm2d(32),
        nn.ReLU(inplace=True),
        nn.Conv2d(32, 64, kernel_size=1, padding=0),
        nn.BatchNorm2d(64),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2)
    ]
    student_submodel = nn.Sequential(*student_block)
    inter_img2 = student_submodel(torch.randn(2,3,32,32))
    print("The size of the student subnet output: ", inter_img2.shape)
    assert inter_img1.shape == inter_img2.shape

    # ===================================================================
    print("# of params in teacher submodel:", get_sum_params(teacher_submodel))
    print("# of params in student submodel:", get_sum_params(student_submodel))

