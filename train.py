import os
import sys
import importlib
import argparse
import math
import torch.nn.functional as F
from models import resnet
from models.utils import train_generater,my_test,correct_cal,result_pre 
import json
import torch
from torchvision import transforms
from torch.utils.data import Dataset
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.nn as nn
from models import utils
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))


if __name__ == "__main__":
    tr_transformer = transforms.Compose([
		transforms.RandomCrop(32, padding=4),  #先四周填充0，在吧图像随机裁剪成32*32
		transforms.RandomHorizontalFlip(),  #图像一半的概率翻转，一半的概率不翻转
		transforms.ToTensor(),
		transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), #R,G,B每层的归一化用到的均值和方差
	])
    te_transformer = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), #R,G,B每层的归一化用到的均值和方差
	])

    training_data = datasets.CIFAR10(
        root="data",
        train=True,
        download=True,
        transform=tr_transformer
	    #target_transform=Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1))
	)
    test_data = datasets.CIFAR10(
        root="data",
        train=False,
        download=True,
        transform=te_transformer
	    #target_transform=Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1))
	)
    train_set = DataLoader(training_data, batch_size = 256, shuffle = True)
    test_set = DataLoader(test_data, batch_size=256)    #load dataset
    classifier = resnet.resnet18().to(device)
    loss_fn = nn.CrossEntropyLoss()
    lr_c = 1e-1
    optim = torch.optim.SGD(classifier.parameters(), lr_c, momentum=0.9, weight_decay=1e-3)
    epochs = 120
    i = 0
    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        i = i + 1
        if (i > 50):
            lr_c = lr_c / 10;
            print(lr_c)
            optim = torch.optim.SGD(classifier.parameters(), lr_c, momentum=0.9, weight_decay=1e-3)
            i = 0;
        utils.resnet_train(train_set, classifier, loss_fn, optim)
        utils.resnet_test(test_set, classifier, loss_fn)
    print("Classifier Done!")
    torch.save(classifier,'./resnet18_cifar10')

