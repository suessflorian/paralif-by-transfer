#!/bin/bash

set -x

pipenv run python main.py attack --attack fgsm --model resnet18 --dataset fashionMNIST
pipenv run python main.py attack --attack fgsm --model resnet18 --dataset fashionMNIST --lif
pipenv run python main.py attack --attack fgsm --model resnet18 --dataset fashionMNIST --paralif
pipenv run python main.py attack --attack fgsm --model resnet50 --dataset fashionMNIST
pipenv run python main.py attack --attack fgsm --model resnet50 --dataset fashionMNIST --lif
pipenv run python main.py attack --attack fgsm --model resnet50 --dataset fashionMNIST --paralif

pipenv run python main.py attack --attack fgsm --model resnet18 --dataset cifar10
pipenv run python main.py attack --attack fgsm --model resnet18 --dataset cifar10 --lif
pipenv run python main.py attack --attack fgsm --model resnet18 --dataset cifar10 --paralif
pipenv run python main.py attack --attack fgsm --model resnet50 --dataset cifar10
pipenv run python main.py attack --attack fgsm --model resnet50 --dataset cifar10 --lif
pipenv run python main.py attack --attack fgsm --model resnet50 --dataset cifar10 --paralif

pipenv run python main.py scratch --epochs 3 --model resnet50 --dataset cifar10 --paralif

pipenv run python main.py attack --attack fgsm --model resnet18 --dataset cifar100
pipenv run python main.py attack --attack fgsm --model resnet18 --dataset cifar100 --lif
pipenv run python main.py attack --attack fgsm --model resnet18 --dataset cifar100 --paralif
pipenv run python main.py attack --attack fgsm --model resnet50 --dataset cifar100
pipenv run python main.py attack --attack fgsm --model resnet50 --dataset cifar100 --lif
pipenv run python main.py attack --attack fgsm --model resnet50 --dataset cifar100 --paralif

pipenv run python main.py attack --attack deepfool --model resnet18 --dataset fashionMNIST
pipenv run python main.py attack --attack deepfool --model resnet18 --dataset fashionMNIST --lif
pipenv run python main.py attack --attack deepfool --model resnet18 --dataset fashionMNIST --paralif
pipenv run python main.py attack --attack deepfool --model resnet50 --dataset fashionMNIST
pipenv run python main.py attack --attack deepfool --model resnet50 --dataset fashionMNIST --lif
pipenv run python main.py attack --attack deepfool --model resnet50 --dataset fashionMNIST --paralif

pipenv run python main.py scratch --epochs 10 --model resnet50 --dataset cifar100 --paralif

pipenv run python main.py attack --attack deepfool --model resnet18 --dataset cifar100
pipenv run python main.py attack --attack deepfool --model resnet18 --dataset cifar100 --lif
pipenv run python main.py attack --attack deepfool --model resnet18 --dataset cifar100 --paralif
pipenv run python main.py attack --attack deepfool --model resnet50 --dataset cifar100
pipenv run python main.py attack --attack deepfool --model resnet50 --dataset cifar100 --lif
pipenv run python main.py attack --attack deepfool --model resnet50 --dataset cifar100 --paralif


