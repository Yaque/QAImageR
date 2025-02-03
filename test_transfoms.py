
import numpy as np
from PIL import Image
import os
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import math
import copy

number = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
sign = ['+', '-', 'x']
characters = number + sign + ['=', '?']
captcha_length = 5
width = 120
height = 40

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src):
        src2 = self.self_attn(src, src, src)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(nn.functional.relu(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

channel = 3
class MNISTTransformerClassifier(nn.Module):
    def __init__(self, d_model=512, nhead=16, num_encoder_layers=12, num_classes=captcha_length):
        super(MNISTTransformerClassifier, self).__init__()
        self.embedding = nn.Linear(channel * width * height, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = TransformerEncoderLayer(d_model, nhead)
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_encoder_layers)
        self.fc = nn.Linear(d_model, num_classes * len(characters))

    def forward(self, x):
        # print(x.size())
        x = x.view(x.size(0), -1)
        # print(x.size())
        x = self.embedding(x)
        # print(x.size())
        x = self.pos_encoder(x)
        # print(x.size())
        x = self.transformer_encoder(x)
        # print(x.size())
        x = x.mean(dim=0)
        # print(x.size())
        x = self.fc(x)
        # print(x.size())
        return x




# 定义一个组合变换，包括转换为张量和归一化
transform = transforms.Compose([
    transforms.ToTensor(),  # 将PIL Image或numpy.ndarray转换为tensor，并将像素值从[0, 255]缩放到[0, 1]
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 使用ImageNet数据集的均值和标准差
])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

def test(path, file):
    # print(path)
    model = MNISTTransformerClassifier().to(device)        
    model.load_state_dict(torch.load("m/qq7_59.375_qq.pkl"))        
    image = Image.open(path)
        
    label_text = file[:3] + "=?"
    print(label_text)
    label = np.zeros(captcha_length, dtype=np.int64)
    for i, c in enumerate(label_text):
        label[i] = characters.index(c)
        
    image = transform(image).unsqueeze(0)
    label = torch.from_numpy(label).unsqueeze(0)
    print(image.size(), label.size())
    image = image.to(device)
    label = label.to(device)
    output = model(image)
    # print(outputs.size(), labels.size())
    output = output.view(output.size()[0], len(characters), captcha_length)
    # print(outputs.size(), labels.size())
    _, predicted_test = torch.max(output.data, 1)
    print("predicted_test", predicted_test)
    print("==============识别结果：", end='')
    for i, c in enumerate(predicted_test[0]):
        print(characters[c], end='')
    
    print("")
    print("==============标准结果：", label_text)

if __name__ == "__main__":

    # 单纯的验证
    # 定义要遍历的文件夹路径
    folder_path = './images/val'

    # 遍历文件夹中的所有文件和子文件夹
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            # 获取文件的完整路径
            file_path = os.path.join(root, file)
            test(file_path, file)