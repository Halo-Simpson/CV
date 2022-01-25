# 99.450
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import cv2
import numpy as np
from torch.utils.data import DataLoader

max_accuracy = 0.0
# 定义超参数
BATCH_SIZE = 256  # 每批处理的数据
DEVICE = torch.device("cuda")  # 选择训练用硬件
EPOCHS = 300  # 训练数据集轮次

# 构建pipeline
pipeline = transforms.Compose([
    transforms.RandomRotation(10),  # 旋转，范围-10°~10°
    transforms.Resize([32, 32]),  # 缩放
    transforms.RandomCrop([28, 28]),  # 随机裁剪
    # transforms.GaussianBlur(3, sigma=(0.1, 2.0)),
    transforms.ToTensor(),  # 将图片转换成tensor
    transforms.Normalize((0.1370,), (0.3081,))  # 正则化，降低模型复杂度
])

# 加载数据
train_set = datasets.MNIST("data", train=True, download=True, transform=pipeline)
test_set = datasets.MNIST("data", train=False, download=True, transform=pipeline)

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=True)


# # 显示
# with open("./data/MNIST/raw/train-images-idx3-ubyte", "rb") as f:
#     file = f.read()
#
# image1 = [int(str(item).encode('ascii'), 16) for item in file[16: 16 + 784]]
# print(image1)
#
# image1_np = np.array(image1, dtype=np.uint8).reshape(28, 28, 1)
# print(image1_np.shape)
# cv2.imwrite("digit.jpg", image1_np)


# 构建网络模型
class Digit(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=10, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
        self.conv2 = nn.Conv2d(in_channels=10, out_channels=20, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

        self.fc1 = nn.Linear(20 * 14 * 14, 500)  # 20*10*10：输入通道 500：输出通道
        self.fc2 = nn.Linear(500, 10)  # 500：输入通道 10：输出通道
        # self.dropout = nn.Dropout(0.5)

    # 前向传播
    def forward(self, x):
        input_size = x.size(0)  # batch_size
        x = self.conv1(x)  # 输入：batch*1*28*28 输出：batch*10*27*27
        x = F.relu(x)
        x = F.max_pool2d(x, 2, 2)

        x = self.conv2(x)  # 输出：batch*20*13*13
        x = F.relu(x)

        x = x.view(input_size, -1)  # -1：自动计算维度

        x = self.fc1(x)  # 输出：batch*500
        x = F.relu(x)
        x = self.fc2(x)  # 输出：batch*10

        output = F.log_softmax(x, dim=1)  # 计算分类后每个数字的概率值

        return output


# 定义优化器
MODEL = Digit().to(DEVICE)

# OPTIMIZER = optim.Adam(MODEL.parameters())
OPTIMIZER = optim.SGD(MODEL.parameters(), lr=0.05, momentum=0.9)



# 定义训练方法
def train_model(model, device, train_loader, optimizer, epoch):
    # 模型训练
    model.train()
    for batch_index, (data, target) in enumerate(train_loader):
        # 部署到DEVICE上去
        data, target = data.to(device), target.to(device)
        # 梯度初始化为0
        optimizer.zero_grad()
        # 训练后的结果
        output = model(data)
        # 计算损失
        loss = F.cross_entropy(output, target)  # 交叉熵
        # 反向传播
        loss.backward()
        # 参数优化
        optimizer.step()
        if batch_index % 3000 == 0:
            print("Train epoch : {} \t : Loss : {:.6f}".format(epoch, loss.item()))


# 定义测试方法
def test_model(model, device, test_loader):
    global max_accuracy
    # 模型验证
    model.eval()
    # 正确率
    correct = 0.0
    # 测试损失
    test_loss = 0.0
    with torch.no_grad():
        for data, target in test_loader:
            # 部署到device上
            data, target = data.to(device), target.to(device)
            # 测试数据
            output = model(data)
            # 计算测试损失
            test_loss += F.cross_entropy(output, target).item()
            # 找到概率最大的下标
            pred = output.max(1, keepdim=True)[1]
            # 累计正确率
            correct += pred.eq(target.view_as(pred)).sum().item()
        test_loss /= len(test_loader.dataset)
        print("Test averager loss : {:.4f}, accuracy : {:.3f}\n".format(
            test_loss, 100.0 * correct / len(test_loader.dataset)))

        if correct / len(test_loader.dataset) > max_accuracy:
            max_accuracy = correct / len(test_loader.dataset)
            print("max_accuracy : {:.3f}".format(max_accuracy))
            torch.save(Digit, 'res')


# 调用上述方法
for epoch in range(1, EPOCHS + 1):
    train_model(MODEL, DEVICE, train_loader, OPTIMIZER, epoch)
    test_model(MODEL, DEVICE, test_loader)
