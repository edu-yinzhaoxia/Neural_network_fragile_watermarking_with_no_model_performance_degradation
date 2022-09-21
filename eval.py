import torch
import numpy as np
from models import resnet
from models.utils import train_generater,my_test,correct_cal,result_pre
from torchvision import transforms
from torch.utils.data import Dataset
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.nn as nn
from models import utils
import matplotlib.pyplot as plt


secret_key = torch.load('./fragile_sample_labels')
fragile_samples = torch.load('./fragile_samples')
target_model = torch.load('./resnet18_cifar10')
if __name__ == '__main__':

    tr_transformer = transforms.Compose([
		transforms.RandomCrop(32, padding=4),  #先四周填充0，在吧图像随机裁剪成32*32
		transforms.RandomHorizontalFlip(),  #图像一半的概率翻转，一半的概率不翻转
		transforms.ToTensor(),
		transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), #R,G,B每层的归一化用到的均值和方差
	])

    training_data = datasets.CIFAR10(
        root="data",
        train=True,
        download=True,
        transform=tr_transformer
	)

    train_set = DataLoader(training_data, batch_size = 256, shuffle = True)
    target_model.eval()
    my_test(fragile_samples,target_model)
    loss_fn = nn.CrossEntropyLoss()
    lr_c = 1e-3
    optim = torch.optim.SGD(target_model.parameters(), lr_c, momentum=0.9, weight_decay=1e-3)
    epochs = 40
    i = 0
    recorrect = []
    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        i = i + 1
        utils.resnet_train(train_set, target_model, loss_fn, optim)
        recorrect.append(correct_cal(secret_key, my_test(fragile_samples, target_model)))
    recorrect = np.asarray(recorrect)
    np.save('./correct',recorrect)



