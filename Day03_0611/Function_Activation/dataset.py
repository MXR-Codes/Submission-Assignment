# 导入操作系统相关功能模块
import os
# 从PyTorch工具包中导入数据相关模块
from torch.utils import data
# 导入Python图像处理库PIL中的Image模块
from PIL import Image

# 定义一个继承自PyTorch Dataset类的图像文本数据集类
class ImageTxtDataset(data.Dataset):
    # 类的初始化方法
    def __init__(self, txt_path: str, folder_name, transform):
        # 保存图像预处理变换函数
        self.transform = transform
        # 获取文本文件所在目录路径
        self.data_dir = os.path.dirname(txt_path)
        # 初始化存储图像路径的空列表
        self.imgs_path = []
        # 初始化存储标签的空列表
        self.labels = []
        # 保存文件夹名称
        self.folder_name = folder_name
        # 以只读方式打开文本文件
        with open(txt_path, 'r') as f:
            # 读取文件所有行
            lines = f.readlines()
        # 遍历每一行数据
        for line in lines:
            # 分割每行数据为图像路径和标签
            img_path, label = line.split()
            # 去除标签前后空格并转换为整数
            label = int(label.strip())
            # 注释掉的代码：拼接完整图像路径（当前未使用）
            # img_path = os.path.join(self.data_dir, self.folder_name, img_path)
            # 将标签添加到标签列表
            self.labels.append(label)
            # 将图像路径添加到路径列表
            self.imgs_path.append(img_path)

    # 返回数据集中样本数量的方法
    def __len__(self):
        return len(self.imgs_path)

    # 根据索引获取单个样本的方法
    def __getitem__(self, i):
        # 获取指定索引的图像路径和标签
        path, label = self.imgs_path[i], self.labels[i]
        # 打开图像文件并转换为RGB格式
        image = Image.open(path).convert("RGB")
        # 如果定义了预处理变换，则应用变换
        if self.transform is not None:
            image = self.transform(image)
        # 返回处理后的图像和对应的标签
        return image, label