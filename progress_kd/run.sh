python vgg16-main.py --role teacher --epochs 1
python vgg16-main.py --role student --epochs 1
python vgg16-main.py --role teacher --epochs 1 --batchnorm
python vgg16-main.py --role student --epochs 1 --batchnorm
