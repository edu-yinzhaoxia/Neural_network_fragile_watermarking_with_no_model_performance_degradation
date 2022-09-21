import torch
import torch.nn as nn
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))
loss_fn = nn.CrossEntropyLoss()#损失函数
soft_max = nn.Softmax(dim=1)

def o_test(dataloader, model):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}% \n")

def train_generater(dataloader, model, optimizerG,target_model, target_label):
    size = len(dataloader)
    num_batches = len(dataloader)
    model.train()
    X= dataloader.to(device)
    pred = target_model(X)
    loss = 1*loss_fn(pred,target_label)+150*torch.mean(torch.var(soft_max(pred)))
    print("此时生成样本在目标分类模型中的标签为：")
    print(torch.var(soft_max(pred)))
    optimizerG.zero_grad()
    loss.backward()
    optimizerG.step()

def my_test(dataloader, model):
    size = len(dataloader)
    num_batches = len(dataloader)
    model.eval()
    test_loss = 0 
    X= dataloader.to(device)
    pred = model(X)
    print(pred.argmax(1))
    return pred.argmax(1)

def result_pre(data,model):
    size = len(data)
    num_batches = len(data)
    model.eval()
    test_loss = 0 
    X= data.to(device)
    pred = model(X)
    print(pred.argmax(1))
    return pred.argmax(1)
    
def correct_cal(first,after):
    i = 0
    length = len(first)
    for x,y in zip(first,after):
        if x==y:
            i = i+1
    return i*100.0/length

def resnet_train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def resnet_test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")