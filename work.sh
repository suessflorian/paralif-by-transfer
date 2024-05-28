#!/bin/bash

set -x

pipenv run python main.py attack --attack square --arch vit_b_16 --dataset cifar100
pipenv run python main.py attack --attack square --arch vit_b_16 --dataset cifar100 --lif
pipenv run python main.py attack --attack square --arch vit_b_16 --dataset cifar100 --paralif
