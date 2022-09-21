from models import pgan,resnet
import math
import torch
from models.utils import train_generater,my_test,correct_cal,result_pre

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))
t = torch.tensor([0,1,2,3,4,5,6,7,8,9,0,1,2,3,4,5,6,7,8,9,0,1,2,3,4,5,6,7,8,9,0,1,2,3,4,5,6,7,8,9,0,1,2,3,4,5,6,7,8,9,0,1,2,3,4,5,6,7,8,9,0,1,2,3,4,5,6,7,8,9,0,1,2,3,4,5,6,7,8,9,0,1,2,3,4,5,6,7,8,9,0,1,2,3,4,5,6,7,8,9])
t = t.to(device)
torch.save(t,'./fragile_sample_labels')

target_model = torch.load('./resnet18_cifar10')
g_model = pgan.Generator(3, 512, 32)  # 生成3维数据，输入1*512维，生成图片大小32*32            #load generator
total_stages = int(math.log2(32 / 4)) + 1
for i in range(total_stages - 1):
    g_model.grow_network()
    g_model.flush_network()
g_model = g_model.to(device)

optimizerG = torch.optim.Adam(g_model.parameters(), lr=1e-4, betas=(0.5, 0.999))
fixed_noise = torch.FloatTensor(100, 512).normal_(0.0, 1.0).to(device)
epoch = 300
for i in range(epoch):  # start
    fake = g_model(fixed_noise)
    c = correct_cal(result_pre(fake, target_model), t)
    train_generater(fake, g_model, optimizerG, target_model, t)
    if (c > 99.9):
        break
my_test(fake, target_model)  # check the label of trigger in target model

torch.save(fake, './fragile_samples')
# 最后保存一下生成的样本fake, 以便后续实验验证效果