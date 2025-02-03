
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
from train import CNN_Network

number = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
sign = ['+', '-', 'x']

# 定义一个组合变换，包括转换为张量和归一化
transform = transforms.Compose([
    transforms.ToTensor(),  # 将PIL Image或numpy.ndarray转换为tensor，并将像素值从[0, 255]缩放到[0, 1]
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 使用ImageNet数据集的均值和标准差
])

def test(path):
    net=CNN_Network()           
    net.load_state_dict(torch.load("27yanzhengma.pkl"))        
    img= cv2.imread(path,0)
    # if img.shape!=(60,160):
    #     cropImage(path)
    #     img= cv2.imread(path,0)
    # cv2.imwrite("1.png",img)
    
    img = img/255.
    # img = transform(img)
    # cv2.imwrite("2.png",img.numpy())
    
    # img = torch.from_numpy(img).float()
    img = torch.unsqueeze(img,0)
    img = torch.unsqueeze(img,0)
    pred = net(img)

    pre1=torch.argmax(pred[:,:15],dim=1)
    pre2 = torch.argmax(pred[:, 15:30], dim=1)
    pre3 = torch.argmax(pred[:, 30:45], dim=1)
    pre4 = torch.argmax(pred[:, 45:60], dim=1)
    pre5 = torch.argmax(pred[:, 60:75], dim=1)
    pred = [pre1, pre2, pre3, pre4, pre5]

    labels=number + sign + ['=', '?']
    for i in pred:
        print(labels[i.item()],end='')
    print("========", path)

if __name__ == "__main__":

    # 定义要遍历的文件夹路径
    folder_path = './images/test'

    # 遍历文件夹中的所有文件和子文件夹
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            # 获取文件的完整路径
            file_path = os.path.join(root, file)
            test(file_path)