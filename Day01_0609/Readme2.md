# 遥感图像处理笔记

## 基本概念

### 遥感图像组成
- 遥感图像由多个**波段(band)**组成，每个波段记录不同光谱范围的信息  
- 常见波段组合：红(Red)、绿(Green)、蓝(Blue)三个波段可合成真彩色图像  

### 图像归一化
- **目的**：将不同范围的像素值映射到统一范围(如0-255)  
- **方法**：线性变换公式 `(x - min) / (max - min) * 255`  
- **作用**：增强图像对比度，便于可视化分析  

## 关键技术点

### 分位数拉伸
- **原理**：使用图像像素值的2%和98%分位数作为最小/最大值进行归一化  
- **优点**：能有效消除极端值影响，突出主体信息  
- **计算步骤**：  
  1. 计算每个波段的2%分位数(`min_val`)和98%分位数(`max_val`)  
  2. 使用线性变换将`[min_val, max_val]`映射到`[0,255]`  

### 图像显示优化
- **数据类型转换**：处理后的浮点数据需转换为8位无符号整数(`np.uint8`)才能正确显示  
- **颜色通道顺序**：RGB图像需要按红、绿、蓝顺序堆叠波段  