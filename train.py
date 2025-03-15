from torchvision.datasets import FashionMNIST,ImageFolder
from torchvision import transforms
import numpy as np
import torch.utils.data as Data
import matplotlib.pyplot as plt
import torch.nn as nn
from model import LeNet,VGG16,Inception,GooglNet,ResNet18,Residual_Block
import time
import pandas as pd
import torch
from torch.utils.data import random_split
import copy
import sys
import os
def data_train_val_Split():
    # train_data = FashionMNIST(
    # root='/root/shared-nvme/chenyq/Practice/DataSets/FashionMNIST',
    # transform=transforms.Compose([transforms.Resize(224),transforms.ToTensor()]),
    # download=False
    # )
    Root='/root/shared-nvme/chenyq/Practice/DataSets/Rice/train'
    normalize = transforms.Normalize([0.687,0.589,0.430],[0.0471, 0.0488, 0.0771])
    train_transforms = transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),normalize])
    train_data = ImageFolder(Root,transform=train_transforms) #ImageFolder可以直接根据train文件夹下子文件名作为类别标签，无需额外标签文件
    # # 获取类别名称
    # class_names = train_data.classes
    # print("Class names:", class_names)
    
    # # 获取类别到索引的映射
    # class_to_idx = train_data.class_to_idx
    # print("Class to index mapping:", class_to_idx)
    # sys.exit()
    total_length = len(train_data)
    train_length = round(0.8 * total_length)
    val_length = total_length - train_length
    train_data, val_data = random_split(train_data, [train_length, val_length])

    train_dataloader = Data.DataLoader(train_data,batch_size=128,shuffle=True,num_workers=8)
    val_dataloader = Data.DataLoader(val_data,batch_size=128,shuffle=True,num_workers=8)
    return train_dataloader,val_dataloader

def trian(model,train_dataloader,val_dataloader,epochs,model_name):
    device = 'cuda' if torch.cuda.is_available else 'cpu'
    optimizer = torch.optim.Adam(model.parameters(),lr=0.005)
    losser = nn.CrossEntropyLoss()
    model = model.to(device)
    best_model_parameters = copy.deepcopy(model.state_dict())

    best_acc = 0.0
    train_acc_all = []
    val_acc_all = []
    train_loss_all = []
    val_loss_all = []

    since = time.time()

    for epoch in range(epochs):
        print("Epoch{}/{}".format(epoch,epochs-1))
        print('-'*10)
        train_loss = 0
        train_acc = 0
        val_loss = 0
        val_acc = 0
        train_num = 0
        val_num = 0
        val_corrects = 0
        train_corrects = 0
        #训练一轮的代码：
        for step,(x,y) in enumerate(train_dataloader):
            image, label = x.to(device), torch.tensor(y).to(device)  
            model.train()
            output = model(image)
            pred = torch.argmax(output,dim=1)
            # print(image.shape)
            # print(pred.shape)
            loss = losser(output,label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()*x.size(0)   #这一步为什么需要*x.size()   答：loss.item()用于获取交叉熵损失函数计算的每个样本的平均值，乘以批次样本数量来表示该批次的总损失（注：交叉熵损失本身就只返回一个值，item()的作用仅仅是将张量转换为python可以直接用的标量，即cpu可以用的变量，避免了过多的梯度计算）

            train_corrects += torch.sum(pred==label)
            train_num += x.size(0)
            # print(train_loss,train_num)
            # print("train_loss:{:.4f},train_corrects{},train_num:{}".format(train_loss,train_corrects,train_num))
            
            
        for step,(x,y) in enumerate(train_dataloader):
            image,label = x.to(device),torch.tensor(y).to(device)
            model.eval()
            output = model(image)
            pred = torch.argmax(output,dim=1)
            loss = losser(output,label)
            val_loss += loss.item()*x.size(0)   #这一步为什么需要*x.size()
            val_corrects += torch.sum(pred==label)
            val_num += x.size(0)
        # print(train_loss)
        # print(train_num)
        # print('---------------------')
        # sys.exit()
        train_loss /= train_num
        val_loss /= val_num
        train_loss_all.append(train_loss)
        val_loss_all.append(val_loss)

        train_acc = train_corrects.double().item() / train_num
        val_acc = val_corrects.double().item() / val_num
        train_acc_all.append(train_acc)
        val_acc_all.append(val_acc)

        print("Epoch:{} train loss:{:.4f} train acc:{:.4f}".format(epoch,train_loss_all[-1],train_acc_all[-1]))
        print("Epoch:{} val loss:{:.4f} val acc:{:.4f}".format(epoch,val_loss_all[-1],val_acc_all[-1]))

        if val_acc_all[-1]>best_acc:
            best_acc = val_acc_all[-1]
            best_model_wts = copy.deepcopy(model.state_dict())#获取最优模型参数
            model_pt_folder = '/root/shared-nvme/chenyq/Practice/tranditionalNet/'+model_name
            if not os.path.exists(model_pt_folder):
                os.makedirs(model_pt_folder)
            model_pt_file = model_pt_folder + '/best.pth'
            torch.save(best_model_wts, model_pt_file) #保存最优模型参数,更换模型时，需要更改保存路径
            
        time_use = time.time()-since  #训练耗时
        print("训练耗时：{:.0f}m{:.0f}s".format(time_use//60,time_use%60))
    model.load_state_dict(best_model_wts) #加载最优模型参数，为下一轮训练做准备

    train_process = pd.DataFrame(data={
        "epoch":range(epochs),
        "train_loss_all":train_loss_all,
        "val_loss_all":val_loss_all,
        "train_acc_all":train_acc_all,
        "val_acc_all":val_acc_all
    })
    return train_process

# def plot_acc_loss(train_preocess):
#     plt.figure(figsize=(12,4))
#     plt.subplot(1,2,1)
#     plt.plot(train_preocess["epoch"],train_preocess.train_loss_all,'ro-',label="train loss")
#     plt.plot(train_preocess["epoch"],train_preocess.val_loss_all,'ro-',label="val loss")
#     plt.legend()
#     plt.xlabel("epoch")
#     plt.ylabel("loss")

#     plt.subplot(1,2,2)
#     plt.plot(train_preocess["epoch"],train_preocess.train_acc_all,'ro-',label="train loss")
#     plt.plot(train_preocess["epoch"],train_preocess.val_acc_all,'ro-',label="val loss")
#     plt.legend()
#     plt.xlabel("epoch")
#     plt.ylabel("acc")
    
#     # 保存图像到当前目录下
#     plt.savefig("VGG_acc_loss.png")  # 保存为 PNG 格式
def plot_acc_loss(train_process):
    plt.figure(figsize=(12, 4))

    # 绘制损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(train_process["epoch"], train_process["train_loss_all"], 'r-o', label="Train Loss")  # 红色圆圈
    plt.plot(train_process["epoch"], train_process["val_loss_all"], 'b-s', label="Val Loss")    # 蓝色正方形
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    # 绘制准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(train_process["epoch"], train_process["train_acc_all"], 'g-^', label="Train Acc")  # 绿色三角形
    plt.plot(train_process["epoch"], train_process["val_acc_all"], 'y-*', label="Val Acc")      # 黄色星号
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")

    # 保存图像到当前目录下
    plt.savefig("ResNet18_acc_loss.png")  # 保存为 PNG 格式

if __name__=="__main__":
    model_name = 'ResNet18-furuit'
    model = ResNet18(Residual_Block)
    epochs = 50
    train_dataloader,val_dataloader = data_train_val_Split()
    train_process = trian(model,train_dataloader,val_dataloader,epochs,model_name)
    plot_acc_loss(train_process)
    
            
            
        
        
        

            
            
        
    
    