import torch
import torch.utils.data as Data
import torchvision
from torchvision import transforms
from torchvision.datasets import FashionMNIST
from model import LeNet

# def data_test_Split():
#     test_data = FashionMNIST(
#     root='/root/shared-nvme/chenyq/Practice/DataSets',
#     train=False,
#     transform=transforms.Compose([transforms.Resize(224),transforms.ToTensor()]),
#     download=True
#     )
    
#     test_dataloader = Data.DataLoader(test_data,batch_size=1,shuffle=True,num_workers=0)
    
#     return test_dataloader
def data_test_Split():
    Root='/root/shared-nvme/chenyq/Practice/DataSets/CatAndDog/train'
    normalize = transforms.Normalize([0.162,0.151,0.138],[0.058,0.052,0.048])
    test_transforms = transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),normalize])
    test_data = ImageFolder(Root,transform=test_transforms)
    
    test_dataloader = Data.DataLoader(test_data,batch_size=1,shuffle=True,num_workers=0)
    
    return test_dataloader

def test(model,test_dataloader,classes):
    device = 'cuda' if torch.cuda.is_available else 'cpu'
    model = model.to(device)

    test_correct = 0
    test_num = 0
    test_acc = 0
    with torch.no_grad():
        for step,(x,y) in enumerate(test_dataloader):
            image = x.to(device)
            # label = torch.tensor(y).to(device)  
            label = y.to(device)
            model.eval()
            output = model(image)
            pred = torch.argmax(output,dim=1)
            print("预测值：{}------>真实值：{}".format(classes[pred],classes[label))
            test_correct +=torch.sum(pred==label)
            test_num += x.size(0)
        test_acc = test_correct.double().item() / test_num
        print("test acc:{:.4f}".format(test_acc))
    # return test_acc
if __name__=="__main__":
    classes = ['cat','dog']
    model = LeNet()
    model.load_state_dict(torch.load('/root/shared-nvme/chenyq/Practice/Class2/LeNetmodel_pt/best.pth'))
    test_dataloader = data_test_Split()
    test(model,test_dataloader,classes)
    
        