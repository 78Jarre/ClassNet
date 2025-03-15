import torch
from torch import nn
from torchsummary import summary

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet,self).__init__()
        self.c1 = nn.Conv2d(in_channels=1,out_channels=6,kernel_size=5,padding=2)
        self.sig = nn.Sigmoid()
        self.p1 = nn.AvgPool2d(kernel_size=2,stride=2)
        self.c2 = nn.Conv2d(in_channels=6,out_channels=16,kernel_size=5,padding=0)
        self.p2 = nn.AvgPool2d(kernel_size=2,stride=2)
        self.flatten = nn.Flatten()  #展平操作
        self.l1 = nn.Linear(46656,1200)
        self.l2 = nn.Linear(1200,120)
        self.l3 = nn.Linear(120,10)

    def forward(self,x):
        # print(x.shape)
        x = self.sig(self.c1(x)) #6*224*224
        # print(x.shape)
        x = self.p1(x)           #6*113*113
        # print(x.shape)
        x = self.sig(self.c2(x)) #16*111*111
        # print(x.shape)
        x = self.p2(x)
        # print(x.shape)
        x = self.flatten(x)
        # print(x.shape)
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        return x

class VGG16(nn.Module):
    def __init__(self):
        super(VGG16,self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=1,out_channels=64,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=128,out_channels=128,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2)
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(in_channels=128,out_channels=256,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256,out_channels=256,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256,out_channels=256,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2)
        )
        self.block4 = nn.Sequential(
            nn.Conv2d(in_channels=256,out_channels=512,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2)
        )
        self.block5 = nn.Sequential(
            nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2),
        )
        self.block6 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(7*7*512,4096),
            nn.ReLU(),
            nn.Linear(4096,4096),
            nn.ReLU(),
            nn.Linear(4096,10)
        )
        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias,0)  #将偏置项置为0
            elif isinstance(m,nn.Linear):
                nn.init.normal_(m.weight,0,0.01) #正态分布初始化（均值为0，方差为0.01）
                if m.bias is not None:
                    nn.init.constant_(m.bias,0)  #将偏置项置为0
        self.Blocks = nn.Sequential(
            self.block1,
            self.block2,
            self.block3,
            self.block4,
            self.block5,
            self.block6
        )

    def forward(self,x):
        x = self.Blocks(x)
        # x = self.block1(x)
        # x = self.block2(x)
        # x = self.block3(x)
        # x = self.block4(x)
        # x = self.block5(x)
        # x = self.block6(x)
        return x
class Inception(nn.Module):
    def __init__(self,in_channels,c1,c2,c3,c4):
        super(Inception,self).__init__()
        self.ReLU = nn.ReLU()
        # 路径1
        self.p1_1 = nn.Conv2d(in_channels=in_channels,out_channels=c1,kernel_size=1)

        # 路径2
        self.p2_1 = nn.Conv2d(in_channels=in_channels,out_channels=c2[0],kernel_size=1)
        self.p2_2 = nn.Conv2d(in_channels=c2[0],out_channels=c2[1],kernel_size=3,padding=1)

        # 路径3
        self.p3_1 = nn.Conv2d(in_channels=in_channels,out_channels=c3[0],kernel_size=1)
        self.p3_2 = nn.Conv2d(in_channels=c3[0],out_channels=c3[1],kernel_size=5,padding=2)
        
        # 路径4
        self.p4_1 = nn.MaxPool2d(kernel_size=3,padding=1,stride=1)
        self.p4_2 = nn.Conv2d(in_channels=in_channels,out_channels=c4,kernel_size=1)

    def forward(self,x):
        p1 = self.ReLU(self.p1_1(x))
        p2 = self.ReLU(self.p2_2(self.ReLU(self.p2_1(x))))
        p3 = self.ReLU(self.p3_2(self.ReLU(self.p3_1(x))))
        p4 = self.ReLU(self.p4_2(self.p4_1(x)))
        # print(p1.shape,p2.shape,p3.shape,p4.shape)
        return torch.cat((p1,p2,p3,p4),dim=1)

class GooglNet(nn.Module):
    def __init__(self,Inception):
        super(GooglNet,self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=64,kernel_size=7,stride=2,padding=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=64,out_channels=64,kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64,out_channels=192,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        )
        self.block3 = nn.Sequential(
            Inception(192,64,(96,128),(16,32),32),
            Inception(256,128,(128,192),(32,96),64),
            nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        )
        self.block4 = nn.Sequential(
            Inception(480,192,(96,208),(16,48),64),
            Inception(512,160,(112,224),(24,64),64),
            Inception(512,128,(128,256),(24,64),64),
            Inception(512,112,(128,288),(32,64),64),
            Inception(528,256,(160,320),(32,128),128),
            nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        )
        self.block5 = nn.Sequential(
            Inception(832,256,(160,320),(32,128),128),
            Inception(832,384,(192,384),(48,128),128),
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(1024,2)
        )
        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,mode="fan_out",nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias,0)
            elif isinstance(m,nn.Linear):
                nn.init.normal_(m.weight,0,0.01)
    def forward(self,x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        return x

class Residual_Block(nn.Module):
    def __init__(self,input_channels,out_channels,use_1conv=False,stride=1):
        super(Residual_Block,self).__init__()
        self.ReLU = nn.ReLU()
        self.conv1 = nn.Conv2d(in_channels=input_channels,out_channels=out_channels,kernel_size=3,padding=1,stride=stride)
        self.conv2 = nn.Conv2d(in_channels=out_channels,out_channels=out_channels,kernel_size=3,padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if use_1conv:
            self.conv3=nn.Conv2d(in_channels=input_channels,out_channels=out_channels,kernel_size=1,stride=stride)
        else:
            self.conv3=None
    def forward(self,x):
        y = self.ReLU(self.bn1(self.conv1(x)))
        # print('-----')
        # print(y.shape)
        # print(x.shape)
        y = self.bn2(self.conv2(y))
        if self.conv3:
            x = self.conv3(x)
        
        y = self.ReLU(y + x)
        return y

class ResNet18(nn.Module):
    def __init__(self,Residual_Block):
        super(ResNet18,self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=64,kernel_size=7,stride=2,padding=3),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        )
        self.block2 = nn.Sequential(
            Residual_Block(64,64,use_1conv=False,stride=1),
            Residual_Block(64,64,use_1conv=False,stride=1),
        )
        self.block3 = nn.Sequential(
            Residual_Block(64,128,use_1conv=True,stride=2),
            Residual_Block(128,128,use_1conv=False,stride=1),
        )
        self.block4 = nn.Sequential(
            Residual_Block(128,256,use_1conv=True,stride=2),
            Residual_Block(256,256,use_1conv=False,stride=1),
        )
        self.block5 = nn.Sequential(
            Residual_Block(256,512,use_1conv=True,stride=2),
            Residual_Block(512,512,use_1conv=False,stride=1),
        )
        self.block6 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(512,5)
        )
    def forward(self,x):
        x = self.block1(x)
        # print(x.shape)
        x = self.block2(x) 
        # print(x.shape)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        return x

    
if __name__=="__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNet18().to(device)
    # model = GooglNet(Inception).to(device)
    print(summary(model,(3,224,224)))
        