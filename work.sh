#!/bin/bash

set -x

pipenv run python main.py train --model resnet18 --dataset cifar100 --lif
pipenv run python main.py train --model resnet18 --dataset cifar100 --paralif
pipenv run python main.py train --model resnet50 --dataset cifar100 --lif
pipenv run python main.py train --model resnet50 --dataset cifar100 --paralif

pipenv run python main.py train --model resnet18 --dataset fashionMNIST --lif
pipenv run python main.py train --model resnet18 --dataset fashionMNIST --paralif
pipenv run python main.py train --model resnet50 --dataset fashionMNIST --lif
pipenv run python main.py train --model resnet50 --dataset fashionMNIST --paralif
