"""
Script to download the cifar 10 data
"""
import torchvision
torchvision.datasets.CIFAR10(root="../data", download=True)