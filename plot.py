from torchvision.datasets import FashionMNIST
from torchvision import transforms
import numpy as np
import torch.utils.data as Data
import matplotlib.pyplot as plt


#下载数据到指定位置,并给出单个数据的获取方式
train_data = FashionMNIST(
    root='/root/shared-nvme/chenyq/Practice/DataSets/FashionMNIST',
    transform=transforms.Compose([transforms.Resize(224),transforms.ToTensor()]),
    download=False
)

train_dataLoader = Data.DataLoader(train_data,batch_size=64,shuffle=True) 

for step,(b_x,b_y) in enumerate(train_dataLoader):
    if step > 0:
        break
batch_x = b_x.squeeze().numpy()
batch_y = b_y.numpy()
class_label = train_data.classes

plt.figure(figsize=(12,5))
for i in np.arange(len(batch_y)):
    plt.subplot(4,16,i+1)
    plt.imshow(batch_x[i,:,:],cmap=plt.cm.gray)
    plt.title(class_label[batch_y[i]],size=10)
    plt.axis("off")
    plt.subplots_adjust(wspace=0.05)
# 保存图像到当前目录下
plt.savefig("image_plot.png")  # 保存为 PNG 格式

