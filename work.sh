#!/bin/bash

set -x

pipenv run python main.py attack --attack square@0.1 --model resnet18 --dataset cifar10
pipenv run python main.py attack --attack square@0.1 --model resnet18 --dataset cifar10 --lif
pipenv run python main.py attack --attack square@0.1 --model resnet18 --dataset cifar10 --paralif
pipenv run python main.py attack --attack square@0.1 --model resnet50 --dataset cifar10
pipenv run python main.py attack --attack square@0.1 --model resnet50 --dataset cifar10 --lif
pipenv run python main.py attack --attack square@0.1 --model resnet50 --dataset cifar10 --paralif
