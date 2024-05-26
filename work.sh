#!/bin/bash

set -x

pipenv run python main.py attack --attack square --arch resnet18 --dataset cifar100
pipenv run python main.py attack --attack square --arch resnet18 --dataset cifar100 --lif
pipenv run python main.py attack --attack square --arch resnet18 --dataset cifar100 --paralif

pipenv run python main.py attack --attack square --arch resnet50 --dataset cifar100
pipenv run python main.py attack --attack square --arch resnet50 --dataset cifar100 --lif
pipenv run python main.py attack --attack square --arch resnet50 --dataset cifar100 --paralif
