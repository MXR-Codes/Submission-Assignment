# 完整的模型训练套路(以CIFAR10为例)
# alex_train2.py
import time

import torch.optim
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from alex import *

# 准备数据集
train_data = torchvision.datasets.CIFAR10(root="../dataset_chen",
                                          train=True,
                                          transform=torchvision.transforms.ToTensor(),
                                          download=True)

test_data = torchvision.datasets.CIFAR10(root="../dataset_chen",
                                         train=False,
                                         transform=torchvision.transforms.ToTensor(),
                                         download=True)

# 数据集长度
train_data_size = len(train_data)
test_data_size = len(test_data)
print(f"训练数据集的长度{train_data_size}")
print(f"测试数据集的长度{test_data_size}")

# 加载数据集
train_loader = DataLoader(train_data, batch_size=64)
test_loader = DataLoader(test_data, batch_size=64)

# 创建模型
ma = alex()
if torch.cuda.is_available():
    ma = ma.cuda()

# 创建损失函数
loss_fn = nn.CrossEntropyLoss()
if torch.cuda.is_available():
    loss_fn = loss_fn.cuda()

# 优化器
# learning_rate = 1e-2 相当于(10)^(-2)
learning_rate = 0.01
optim = torch.optim.SGD(ma.parameters(), lr=learning_rate)

# 设置训练网络的一些参数
total_train_step = 0  # 记录训练的次数
total_test_step = 0  # 记录测试的次数
epoch = 20  # 训练的轮数

# 添加tensorboard
writer = SummaryWriter("../logs_train")

# 添加开始时间
start_time = time.time()

for i in range(epoch):
    print(f"-----第{i + 1}轮训练开始-----")
    # 训练步骤
    for data in train_loader:
        imgs, targets = data
        if torch.cuda.is_available():
            imgs = imgs.cuda()
            targets = targets.cuda()
        outputs = ma(imgs)
        loss = loss_fn(outputs, targets)

        # 优化器优化模型

        optim.zero_grad()  # 梯度清零
        loss.backward()  # 反向传播
        # 梯度更新
        optim.step()

        total_train_step += 1
        if total_train_step % 500 == 0:
            print(f"第{total_train_step}的训练的loss:{loss.item()}")
            writer.add_scalar("train_loss", loss.item(), total_train_step)

    # 测试步骤（以测试数据上的正确率来评估模型）
    total_test_loss = 0.0
    # 整体正确个数
    total_accuracy = 0
    with torch.no_grad():
        for data in test_loader:
            imgs, targets = data
            if torch.cuda.is_available():
                imgs = imgs.cuda()
                targets = targets.cuda()
            outputs = ma(imgs)
            # 损失
            loss = loss_fn(outputs, targets)
            total_test_loss += loss.item()
            # 正确率
            accuracy = (outputs.argmax(1) == targets).sum()
            total_accuracy += accuracy

    print(f"整体测试集上的loss:{total_test_loss}")
    print(f"整体测试集上的正确率：{total_accuracy / test_data_size}")
    writer.add_scalar("test_loss", total_test_loss, total_test_step)
    writer.add_scalar("test_accuracy", total_accuracy, total_test_step)
    total_test_step += 1

    # 保存最后一轮训练模型
    if i == epoch-1:
        torch.save(ma, f"model_save_alex\\ma_{i}.pth")
        print("模型已保存")

writer.close()

'''
训练数据集的长度50000
测试数据集的长度10000
-----第1轮训练开始-----
第500的训练的loss:2.2799837589263916
整体测试集上的loss:331.9819201231003
整体测试集上的正确率：0.24300000071525574
-----第2轮训练开始-----
第1000的训练的loss:2.0343739986419678
第1500的训练的loss:1.939653754234314
整体测试集上的loss:319.7589852809906
整体测试集上的正确率：0.2671999931335449
-----第3轮训练开始-----
第2000的训练的loss:2.1821634769439697
整体测试集上的loss:304.94711196422577
整体测试集上的正确率：0.3166999816894531

-----第20轮训练开始-----
第15000的训练的loss:0.7829159498214722
第15500的训练的loss:0.905712366104126
整体测试集上的loss:162.84622192382812
整体测试集上的正确率：0.651199996471405
'''