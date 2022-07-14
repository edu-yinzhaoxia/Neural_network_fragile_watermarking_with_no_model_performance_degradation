import torch
import torch.nn as nn
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))
loss_fn = nn.CrossEntropyLoss()#损失函数
t = torch.tensor([0,1,2,3,4,5,6,7,8,9,0,1,2,3,4,5,6,7,8,9,0,1,2,3,4,5,6,7,8,9,0,1,2,3,4,5,6,7,8,9,0,1,2,3,4,5,6,7,8,9,0,1,2,3,4,5,6,7,8,9,0,1,2,3,4,5,6,7,8,9,0,1,2,3,4,5,6,7,8,9,0,1,2,3,4,5,6,7,8,9,0,1,2,3,4,5,6,7,8,9])
t = t.to(device)
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

def train_generater(dataloader, model,target_model):
    size = len(dataloader)
    num_batches = len(dataloader)
    model.train()
    X= dataloader.to(device)
    pred = target_model(X)
    loss = 1*loss_fn(pred,t)+1*torch.mean(torch.var(soft_max(pred)))
    print(soft_max(pred))
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