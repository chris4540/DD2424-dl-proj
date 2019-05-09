#!/bin/bash
python vgg16-main.py --role teacher --epochs 60
python vgg16-main.py --role student --epochs 60
python vgg16-main.py --role teacher --epochs 30 --batchnorm
python vgg16-main.py --role student --epochs 30 --batchnorm
