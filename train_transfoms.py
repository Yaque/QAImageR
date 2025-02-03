import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
# from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
import math
import copy
import os
from PIL import Image
import numpy as np

number = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
sign = ['+', '-', 'x']

# characters = string.digits + string.ascii_uppercase
characters = number + sign + ['=', '?']

# 验证码长度
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

class CaptchaDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = os.listdir(root_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir, self.images[idx])
        # image = Image.open(img_name).convert('L')  # 转换为灰度图
        image = Image.open(img_name)
        
        label_text = self.images[idx][:3] + "=?"
        label = np.zeros(captcha_length, dtype=np.int64)
        for i, c in enumerate(label_text):
            label[i] = characters.index(c)
            
        if self.transform:
            image = self.transform(image)

        return image, label

# 训练 算式验证码识别

# 定义一个组合变换，包括转换为张量和归一化
transform = transforms.Compose([
    transforms.Resize((120,40)),
    transforms.ToTensor(),  # 将PIL Image或numpy.ndarray转换为tensor，并将像素值从[0, 255]缩放到[0, 1]
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 使用ImageNet数据集的均值和标准差
])

bs = 64
train_dataset = CaptchaDataset(root_dir='images/Train', transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=bs, shuffle=True)

test_dataset = CaptchaDataset(root_dir='images/Test', transform=transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=bs, shuffle=True)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

model = MNISTTransformerClassifier().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.000003)

writer = SummaryWriter('runs/qya_transformer-4')

num_epochs = 8

for epoch in range(num_epochs):
    running_loss = 0.0
    correct_train = 0
    total_train = 0
    # 训练阶段
    model.train()  # 确保处于训练模式
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        # print(outputs.size(), labels.size(), labels)
        outputs = outputs.view(outputs.size()[0], len(characters), captcha_length)
        # print(outputs.size(), outputs.type(), labels.size(), labels.type())
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        
        # print(outputs.size())
        _, predicted_train = torch.max(outputs.data, 1)
        # print("yu",predicted_train.size(), labels.size(), predicted_train, labels)
        total_train += labels.size(0)
        deng = (predicted_train == labels).all(dim=1)
        # print("deng:",deng)
        deng = deng.sum()
        # print("deng:",deng)
        deng = deng.item()
        # print("deng:",deng)
        correct_train += deng

        if (i + 1) % 20 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {running_loss / 100}')
            writer.add_scalar('Training Loss', running_loss / 100, epoch * len(train_loader) + i)
            running_loss = 0.0
    print("correct_train", correct_train, "total_train", total_train)
    train_accuracy = 100 * correct_train / total_train
    print(f'Epoch {epoch + 1}: Training Accuracy: {train_accuracy}%')
    writer.add_scalar('Training Accuracy', train_accuracy, epoch)
    
    # 每个epoch结束后，在测试集上评估准确率
    model.eval()  # 设置为评估模式
    correct_test = 0
    total_test = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            # print(outputs.size(), labels.size())
            outputs = outputs.view(outputs.size()[0], len(characters), captcha_length)
            # print(outputs.size(), labels.size())
            _, predicted_test = torch.max(outputs.data, 1)
            # print(predicted_test.size(), labels.size())
            total_test += labels.size(0)
            correct_test += (predicted_test == labels).all(dim=1).sum().item()
    test_accuracy = 100 * correct_test / total_test
    print(f'Epoch {epoch + 1}: Test Accuracy: {test_accuracy}%')
    writer.add_scalar('Test Accuracy', test_accuracy, epoch)
    model.train()  # 切换回训练模式
    torch.save(model.state_dict(), "m/qq" + str(epoch)+ "_" + str(test_accuracy) + "_qq.pkl")

writer.close()
