# 导入必要的库
import numpy as np  # 用于数值计算
import rasterio  # 用于读写地理空间栅格数据
import matplotlib  # 绘图库
import matplotlib.pyplot as plt  # matplotlib的绘图接口

# 设定支持中文字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 设置默认字体为SimHei(黑体)
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

def normalize_band(band, min_val=0, max_val=10000):
    """ 归一化波段数据到 0-255 """
    band = np.clip(band, min_val, max_val)  # 将数据限制在[min_val, max_val]范围内
    return ((band - min_val) / (max_val - min_val)) * 255  # 线性归一化到0-255范围

def shuchu(tif_file):
    """ 读取原始图像并转换为 RGB """
    with rasterio.open(tif_file) as src:  # 打开TIFF文件
        bands = src.read()  # 读取所有波段数据，假设波段顺序为 B02, B03, B04, B08, B12
        # 对红(2)、绿(1)、蓝(0)三个波段进行归一化处理
        red, green, blue = map(normalize_band, [bands[2], bands[1], bands[0]])
        # 将三个波段叠加形成RGB图像，并转换为8位无符号整数格式
        rgb_image = np.stack([red, green, blue], axis=-1).astype(np.uint8)

    return rgb_image  # 返回原始 RGB 图像

def process_remote_sensing_image(input_path):
    """ 处理遥感图像并显示对比结果 """
    with rasterio.open(input_path) as src:  # 打开输入图像文件
        data = src.read().astype(np.float32)  # 读取所有波段数据并转换为浮点型

        # 计算 2% 和 98% 分位数拉伸
        min_vals = np.nanpercentile(data, 2, axis=(1, 2))  # 计算每个波段2%分位数
        max_vals = np.nanpercentile(data, 98, axis=(1, 2))  # 计算每个波段98%分位数
        range_vals = max_vals - min_vals  # 计算动态范围
        range_vals[range_vals == 0] = 1  # 避免除零错误，将0范围设为1

        # 归一化数据到 0-255
        scaled_data = (data - min_vals[:, None, None]) / range_vals[:, None, None] * 255
        scaled_data = np.clip(scaled_data, 0, 255).astype(np.uint8)  # 限制范围并转换类型

        # 选取 RGB 三个通道(波段2,1,0)
        rgb_data = np.stack([scaled_data[2], scaled_data[1], scaled_data[0]], axis=-1)

    # 创建1行2列的子图，用于显示对比结果
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # 显示原始 RGB 图像
    ax1.imshow(shuchu(input_path))  # 调用shuchu函数获取原始RGB图像
    ax1.axis("off")  # 关闭坐标轴
    ax1.set_title("原始 RGB 图像")  # 设置子图标题

    # 显示处理后图像
    ax2.imshow(rgb_data)  # 显示经过分位数拉伸处理的图像
    ax2.axis("off")  # 关闭坐标轴
    ax2.set_title("处理后 RGB 图像")  # 设置子图标题

    plt.show()  # 显示图像

# 运行代码
input_image = "photo.tif"  # 输入图像文件名
process_remote_sensing_image(input_image)  # 调用处理函数