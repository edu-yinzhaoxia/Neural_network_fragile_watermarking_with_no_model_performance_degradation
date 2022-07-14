import os
import sys
import importlib
import argparse
import math
import torch.nn.functional as F
from models import pgan
from models import resnet
from models.utils import train_generater,my_test,correct_cal,result_pre 
import json
import torch
from torchvision import transforms
from torch.utils.data import Dataset
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.nn as nn


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Testing script')
    parser.add_argument('-m', '--model', help="Model's name",
                        type=str, dest="model", default="default")
    parser.add_argument('-n', '--name', help="Dataset's name",
                        type=str, dest="name", default="default")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using {} device".format(device))
    baseArgs, unknown = parser.parse_known_args()
    #target_model = get_Model(baseArgs.model)
    #print("ok")
    #print(baseArgs.model)
    g_model = pgan.Generator(3,512,32)            #load generator
    total_stages = int(math.log2(32/4)) + 1
    for i in range(total_stages-1):
        g_model.grow_network()
        g_model.flush_network()
    g_model = g_model.to(device)
    target_model = resnet.resnet18().to(device)  #load target model
    target_model.load_state_dict(torch.load("./train_model/resnet18_cifar10"))#该模型的精度在cifar10验证集上是91.*%
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
    test_set = DataLoader(test_data, batch_size=128)    #load dataset
    soft_max = nn.Softmax(dim=1)
    optimizerG = torch.optim.Adam(g_model.parameters(), lr=1e-4, betas=(0.5,0.999))
    fixed_noise = torch.FloatTensor(100, 512).normal_(0.0, 1.0).to(device)
    epoch = 300   
    for i in range(epoch):   #start   
        fake = g_model(fixed_noise) 
        c = correct_cal(result_pre(fake,target_model),t)
        train_generater(fake,g_model,target_model)
        if(c>99.9):
            break
    my_test(fake,target_model)    #check the label of trigger in target model
