
import cv2
from captcha.image import ImageCaptcha
import numpy as np
from PIL import Image
import random
import os
from torch.utils.data import DataLoader,Dataset
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


number = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
sign = ['+', '-', 'x']

def one_hot(text):
    # print(text)
    vector = np.zeros(5*15)  #(10+26+26)*4
    allsign = number + sign + ['=', '?']
    for i, c in enumerate(text):
        idx = i * 15 + allsign.index(c)
        vector[idx] = 1
    return vector

class MyDataSet(Dataset):
    def __init__(self,dir):
        self.dir=dir
        self.img_name= next(os.walk(self.dir))[2]
    def __getitem__(self, index):
        img_path = os.path.join(self.dir,self.img_name[index])
        img = cv2.imread(img_path,0)
        img = img/255.
        img = torch.from_numpy(img).float()
        img = torch.unsqueeze(img,0)
        label = torch.from_numpy(one_hot(self.img_name[index][:3]+'=?')).float()
        return img,label
    def __len__(self):
        return len(self.img_name)

class CNN_Network(nn.Module):
    def __init__(self):
        super(CNN_Network, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, stride=1, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, stride=1, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(stride=2, kernel_size=2),  # 30 80
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, stride=1, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64,128,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,stride=2),   # 15 40
        )

        self.fc = nn.Sequential(
            nn.Linear(128 * 10 * 30, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, 75),
        )
        
    def forward(self, x):
        # print(x.size())
        x = self.layer1(x)
        # print(x.size())
        x = self.layer2(x)
        # print(x.size())
        x = self.layer3(x)
        # print(x.size())
        x = x.view(x.size(0), -1)
        # print(x.size())
        x = self.fc(x)
        return x
    
    
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 160)
        self.fc2 = nn.Linear(160, 75)

        # Spatial transformer localization-network
        self.localization = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 3 * 3, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    # Spatial transformer network forward function
    def stn(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, 10 * 3 * 3)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)

        return x

    def forward(self, x):
        # transform the input
        x = self.stn(x)

        # Perform the usual forward pass
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x   

class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=11, stride=4, padding=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.fc = nn.Linear(768, 75)

    def forward(self, x):
        x = self.features(x)
        # print(x.size())
        x = x.view(x.size(0), -1)
        # print(x.size())
        x = self.fc(x)
        return x


def alexnet(num_classes):
    return AlexNet(num_classes=num_classes)

def train(net, train_iter, test_iter, optimizer, loss,device, num_epochs):
    net = net.to(device)
    train_acc_list,test_acc_list,train_loss_list,test_loss_list=[],[],[],[]
    flag=0.0
    for epoch in range(num_epochs):
        print("Epoch:", epoch)
        train_loss,n=0.0,0
        for X, y in train_iter:
            X = X.to(device)
            y = y.to(device)
            y_hat = net(X)
            # print(y_hat.size(), y.size())
            l = loss(y_hat, y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            train_loss+= l.cpu().item()
            n+=y.shape[0]
        print("train_loss=",train_loss)
        train_acc=get_acc(net,train_iter,device)
        test_acc=get_acc(net,test_iter,device)
        print("test_acc=",test_acc)
        if epoch>=2:
            if test_acc>flag:
                torch.save(net.state_dict(), str(epoch)+ "_" + str(test_acc) + "_yanzhengma.pkl")
        flag=test_acc
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        #train_loss_list.append(train_loss)
    torch.save(net.state_dict(), "yanzhengma.pkl")

    draw_losss(train_loss_list,test_loss_list,num_epochs)
    draw_acc(train_acc_list,test_acc_list,num_epochs)
def get_acc(net,data_iter,device):
    acc_sum,n=0,0
    for X,y in data_iter:
        X=X.to(device)
        y=y.to(device)
        y_hat=net(X)
        pre1=torch.argmax(y_hat[:,:15],dim=1)
        real1=torch.argmax(y[:,:15],dim=1)
        pre2 = torch.argmax(y_hat[:, 15:30], dim=1)
        real2 = torch.argmax(y[:, 15:30], dim=1)
        pre3 = torch.argmax(y_hat[:, 30:45], dim=1)
        real3 = torch.argmax(y[:, 30:45], dim=1)
        pre4 = torch.argmax(y_hat[:, 45:60], dim=1)
        real4 = torch.argmax(y[:, 45:60], dim=1)
        pre5 = torch.argmax(y_hat[:, 60:75], dim=1)
        real5 = torch.argmax(y[:, 60:75], dim=1)
        pre_lable=torch.cat((pre1,pre2,pre3,pre4,pre5),0).view(5,-1)
        real_label=torch.cat((real1,real2,real3,real4,real5),0).view(5,-1)
        bool_=(pre_lable==real_label).transpose(0,1)
        n+=y.shape[0]
        for i in range(0,y.shape[0]):
            if bool_[i].int().sum().item()==4:
                acc_sum+=1
    return acc_sum/n

def draw_losss(train_loss,test_loss,epoch):
    plt.clf()
    x=[i for i in range(epoch)]
    plt.plot(x,train_loss,label='train_loss')
    plt.plot(x,test_loss,label='test_loss')
    plt.legend()
    plt.title("loss goes by epoch")
    plt.xlabel('eopch')
    plt.ylabel('loss_value')
    plt.savefig('loss.png')

def draw_acc(train_acc,test_acc,epoch):
    plt.clf()
    x=[i for i in range(epoch)]
    plt.plot(x,train_acc,label='train_acc')
    plt.plot(x,test_acc,label='test_acc')
    plt.legend()
    plt.title("acc goes by epoch")
    plt.xlabel('eopch')
    plt.ylabel('acc_value')
    plt.savefig('acc.png')
if __name__ == "__main__":
   


    num_epochs=50
    batch_size=512
    lr=0.001

    train_dataset = MyDataSet('./images/train')
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
    test_dataset = MyDataSet('./images/test')
    test_loader = DataLoader(test_dataset, shuffle=True, batch_size=batch_size)

    net = CNN_Network()
    # net = Net()
    # net = AlexNet()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    loss = nn.MultiLabelSoftMarginLoss()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train(net, train_loader, test_loader, optimizer, loss, device, num_epochs)
