# 深度神经网络知识点总结

## 1. 环境配置与工具安装
- 新建conda环境：`conda create -n pytorch python=3.9`
- 激活环境：`conda activate pytorch`
- 检查GPU驱动：`nvidia-smi`
- 验证PyTorch GPU支持：`torch.cuda.is_available()`
- 安装Jupyter：`conda install nb_conda`

## 2. PyTorch基础操作
- **工具箱结构查看**：
  - `dir(pytorch)` 查看主模块结构
  - `dir(pytorch.3)` 查看子模块结构
- **工具使用说明**：
  - `help(pytorch.3.a)` 查看具体函数/类的文档

## 3. 数据处理关键组件
- **Dataset核心功能**：
  1. 实现数据与标签的对应关系
  2. 提供数据总量统计功能
- **Dataloader作用**：
  - 对Dataset数据进行批量打包
  - 支持多进程加载等高级特性

## 4. 模型训练核心流程
- **损失函数(Loss)**：
  - 量化模型预测与真实值的差距
  - 为反向传播提供梯度依据
- **模型保存与加载**：
  - 保存训练好的权重参数
  - 加载模型进行推理或继续训练

## 5. GPU加速训练要点
- **需要迁移到GPU的对象**：
  1. 网络模型：`model.cuda()`
  2. 输入数据：`inputs.cuda()`
  3. 损失函数：`loss_fn.cuda()`
- **训练监控工具**：
  - 使用TensorBoard可视化：`tensorboard --logdir=logs`