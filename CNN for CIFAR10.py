import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms

from PIL import Image

# /Applications/Python\ 3.13/Install\ Certificates.command
# 数据预处理
transform = transforms.Compose([
    transforms.ToTensor(), # 把一张图片从PIL Image或者numpy array转换成Torch Tensor
                           # 像素值[0, 255]  →  [0.0, 1.0]
                           # 格式(H, W, C) → (C, H, W)
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # x∈[0,1]映射到x∈[-1,1]
])

# DataLoader
train_data = torchvision.datasets.CIFAR10(root='./cnn_data', train=True, transform=transform, download=True)
test_data = torchvision.datasets.CIFAR10(root='./cnn_data', train=False, transform=transform, download=True)

train_loaders = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True, num_workers=2)
test_loaders = torch.utils.data.DataLoader(test_data, batch_size=32, shuffle=True, num_workers=2)

image, label = train_data[0]

class_names = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# 定义CNN模型
class Neural_Net(nn.Module):
  def __init__(self):
    super().__init__()
    self.conv1 = nn.Conv2d(3, 16, 5) # (32 - 5 / 1) + 1 <-- (12, 28, 28)
    self.pool = nn.MaxPool2d(2, 2) # (12, 14, 14)
    self.conv2 = nn.Conv2d(16, 32, 5) # (32, 10, 10)
    self.fc1 = nn.Linear(32*5*5, 120)
    self.fc2 = nn.Linear(120, 64)
    self.fc3 = nn.Linear(64, 10)

  def forward(self, x):
    x = F.relu(self.conv1(x))
    x = self.pool(x)
    x = F.relu(self.conv2(x))
    x = self.pool(x)
    x = torch.flatten(x, 1)
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = self.fc3(x)

    return x
  

# 创建模型实例
net = Neural_Net()
losses = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=1e-3, momentum=0.9)


def main():
  # 训练模型
  for epoch in range(80):
    print(f"Training Epoch {epoch}...")
    running_loss = 0.0  # 用来累计当前 epoch 中所有 batch 的 loss。

    for i, data in enumerate(train_loaders):  # 内层 batch 循环
      inputs, labels = data
      # inputs: (B, C, H, W)
      # labels: (B,)
      optimizer.zero_grad()

      outputs = net(inputs)

      loss = losses(outputs, labels)
      loss.backward()
      optimizer.step()

      running_loss += loss.item()

    print(f"Loss = {running_loss/len(train_loaders):.4f}")

  # 保存模型参数
  torch.save(net.state_dict(), 'trained_net_for_CIFAR10.pth')

  # 评估模型
  correct = 0
  total = 0

  net.eval()

  with torch.no_grad():
    for data in test_loaders:
      images, labels = data
      outputs = net(images)
      prediction = torch.argmax(outputs, 1)
      total += labels.size(0)
      correct += (prediction == labels).sum().item()

  accuracy = 100*correct/total
  print(f'Accuracy: {accuracy}%')

  # 测试模型
  # 新模型预处理
  new_transform = transforms.Compose([
      transforms.Resize((32, 32)),
      transforms.Lambda(lambda img: img.convert("RGB")),
      transforms.ToTensor(),
      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
  ])
  # 图片加载
  def load_image(image_path):
    image = new_transform(Image.open(image_path))
    image = image.unsqueeze(0)
    return image
  # 图片路径
  image_path = '/Users/apple/Desktop/Pytorch-tutorial/example1.png'
  image = load_image(image_path)
  # 模型评估
  net.eval()
  with torch.no_grad():
    output = net(image)
    prediction = torch.argmax(output, 1)
    print(f'The image is a picture of {class_names[prediction]}')


if __name__ == '__main__':
  main()
