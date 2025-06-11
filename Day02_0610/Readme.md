# 深度神经网络知识点总结

## 1. 环境配置与工具安装
- 新建conda环境：`conda create -n pytorch python=3.9`
  - 指定Python版本可避免包兼容性问题
  - 推荐使用Python 3.8-3.10稳定版本
- 激活环境：`conda activate pytorch`
  - Windows系统需要使用`activate pytorch`
- 检查GPU驱动：`nvidia-smi`
  - 查看CUDA版本和GPU内存占用情况
  - 确保驱动版本与PyTorch要求的CUDA版本匹配
- 验证PyTorch GPU支持：`torch.cuda.is_available()`
  - 返回True表示GPU可用
  - 可进一步检查设备数量：`torch.cuda.device_count()`
- 安装Jupyter：`conda install nb_conda`
  - 推荐安装JupyterLab：`conda install -c conda-forge jupyterlab`
  - 扩展插件：`pip install jupyter_contrib_nbextensions`

## 2. PyTorch基础操作
- **工具箱结构查看**：
  - `dir(pytorch)` 查看主模块结构
    - 常用子模块：torch, nn, optim, utils
  - `dir(pytorch.3)` 查看子模块结构
    - 可配合`__all__`属性查看公开接口
- **工具使用说明**：
  - `help(pytorch.3.a)` 查看具体函数/类的文档
  - 推荐使用IPython的`?`和`??`快捷查看方式
  - 官方文档地址：https://pytorch.org/docs/stable/

## 3. 数据处理关键组件
- **Dataset核心功能**：
  1. 实现数据与标签的对应关系
    - 需重写`__getitem__`和`__len__`方法
    - 支持索引访问和数据切片
  2. 提供数据总量统计功能
    - 便于计算epoch和batch数量
    - 与Dataloader配合实现自动分批
- **Dataloader重要参数**：
  - `batch_size`: 典型值32/64/128
  - `shuffle`: 训练集必须设为True
  - `num_workers`: 建议设为CPU核心数
  - `pin_memory`: GPU训练时建议启用
- **数据增强技巧**：
  - 使用`torchvision.transforms`模块
  - 常见操作：RandomCrop, RandomHorizontalFlip
  - 自定义转换函数：`lambda`或函数式编程

## 4. 模型训练核心流程
- **损失函数(Loss)**：
  - 分类任务：CrossEntropyLoss, NLLLoss
  - 回归任务：MSELoss, L1Loss
  - 自定义损失：继承`nn.Module`实现
- **优化器配置**：
  - Adam优化器：`optim.Adam(model.parameters(), lr=0.001)`
  - 学习率调度：`torch.optim.lr_scheduler`
  - 梯度裁剪：`torch.nn.utils.clip_grad_norm_`
- **模型保存与加载**：
  - 完整保存：`torch.save(model, path)`
  - 只保存参数：`torch.save(model.state_dict(), path)`
  - 加载检查点：`model.load_state_dict(torch.load(path))`
  - 多GPU训练保存需注意模型包装方式

## 5. GPU加速训练要点
- **设备迁移最佳实践**：
  ```python
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  model.to(device)
  inputs = inputs.to(device)