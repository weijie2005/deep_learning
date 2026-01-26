import torch
import matplotlib.pyplot as plt
import os
'''
一、CNN卷积神经网络：CNN全称Convolutional Neural Network）：主要用于图像分类、视频分类、物体检测等任务。
    1. 输入层：输入图像。
    2. 卷积层CONV(Convolutional Layer)：用于提取图像特征，后跟激活函数ReLU。
    3. 池化层POOL(Pooling Layer)：用于减少特征图的空间维度，用于降维。
    4. 全连接层FC(Fully Connected Layer)：输出层，输出分类结果。


二、CNN卷积层实现方法：
    1.卷积的计算过程：
        1.1 输入图像：[H,W,C]
        1.2 卷积核：[F,F,C]，就相当于权重矩阵，每个卷积核的参数数量为F*F*C+1（最后一个参数为偏置项）。
        1.3 输出特征图：[H_out,W_out,C_out],其中H_out和W_out是输出特征图的高度和宽度，C_out是输出特征图的通道数（卷积核的数量）。
        1.4 卷积计算：
            1.4.1 对输入图像的每个通道，用卷积核进行卷积操作。
            1.4.2 对每个通道的卷积结果，进行逐元素点成再求和。
            1.4.3 对求和结果，加上偏置项。
            1.4.4 对结果应用激活函数。
        1.5 输出特征图的尺寸计算：
            1.5.1 输出特征图的高度H_out=(H-F+2P)/S+1，其中H是输入图像的高度，F是卷积核的高度，P是填充大小，S是步长。
            1.5.2 输出特征图的宽度W_out=(W-F+2P)/S+1，其中W是输入图像的宽度，F是卷积核的宽度，P是填充大小，S是步长。'

    2.卷积的计算：
        1.1 卷积核的滑动：卷积核在输入图像上滑动，每次滑动一个步长，直到卷积核覆盖到输入图像的所有位置。
        1.2 stride：卷积核在输入图像上滑动的步长，步长可以是1或2,默认值为1。步长变大，输出特征图的尺寸变小。
        1.3 元素点乘：卷积核与输入图像上的每个对应位置元素进行点乘。
        1.4 求和：将所有位置的乘法结果求和，得到卷积层的输出特征图上的一个像素值。

    3.padding：填充，用于在输入图像的边界添加额外的像素值，以保持输出特征图的尺寸与输入相同。
        1.1 填充方式：
            1.1.1 零填充（Zero Padding）：在输入图像的边界添加额外的像素值为0。
            1.1.2 反射填充（Reflection Padding）：在输入图像的边界反射像素值。
        
        1.2 填充大小：
            1.2.1 单侧填充：在输入图像的边界添加额外的像素值，填充大小为1。
            1.2.2 双侧填充：在输入图像的边界添加额外的像素值，填充大小为2。
            1.2.3 四侧填充：在输入图像的四周添加额外的像素值，填充大小为4。
        
    4.卷积的超参数：
        1.1 卷积核大小（Filter Size）：卷积核的高度和宽度，通常为3x3或5x5。
        1.2 卷积核数量（Number of Filters）：卷积层中卷积核的数量，也称为输出特征图的通道数。
        1.3 步长（Stride）：卷积核在输入图像上滑动的步长，步长可以是1或2,默认值为1。步长变大，输出特征图的尺寸变小。
        1.4 填充（Padding）：在输入图像的边界添加额外的像素值，以保持输出特征图的尺寸与输入相同。

三、Pooling池化层实现方法：
    1.池化的实现方法：
        1.池化核（Pooling Kernel）：或称池化窗口（Pooling Window），用于提取特征图的局部信息,降维，减少数量大小，提高模型的泛化能力。
            [F,F]，池化核的参数数量为F*F+1（最后一个参数为偏置项）。

        2.最大池化
            1.4.1 对输入特征图的每个通道，用池化核进行池化操作。
            1.4.2 对每个通道的池化结果，取最大值。

        3.平均池化
            1.4.1 对输入特征图的每个通道，用池化核进行池化操作。
            1.4.2 对每个通道的池化结果，取平均值。

    2.池化的计算过程：
        1.1 输入特征图：[H,W,C]
        1.2 池化核：[F,F]，池化核的参数数量为F*F+1（最后一个参数为偏置项）。
        1.3 输出特征图：[H_out,W_out,C]，其中H_out和W_out是输出特征图的高度和宽度，C是输出特征图的通道数（与输入特征图相同）。
        1.4 池化计算：
            1.4.1 对输入特征图的每个通道，用池化核进行池化操作。
            1.4.2 对每个通道的池化结果，取最大值（最大池化）或平均值（平均池化）。
            1.4.3 对结果应用激活函数。
        1.5 输出特征图的尺寸计算：
            1.5.1 输出特征图的高度H_out=(H-F+2P)/S+1，其中H是输入图像的高度，F是池化核的高度，P是填充大小，S是步长。
            1.5.2 输出特征图的宽度W_out=(W-F+2P)/S+1，其中W是输入图像的宽度，F是池化核的宽度，P是填充大小，S是步长。

'''
# 中文显示设置
plt.rcParams["font.sans-serif"]=["SimHei"]
plt.rcParams["axes.unicode_minus"]=False

def mul_demo():
    '''
    点乘（元素-wise multiplication）：对应位置的元素相乘。
    '''
    a = torch.tensor([1, 2, 3])
    b = torch.tensor([4, 5, 6])
    # 方式1：用 * 符号（推荐，简洁）,# 方式2：用 torch.mul() 函数
    res = a * b    
    print("一维张量点乘结果：", res)  # 输出：tensor([ 4, 10, 18])

    # 示例2：二维张量点乘（形状必须一致）
    c = torch.tensor([[1, 2], 
                      [3, 4]]
                    )
    d = torch.tensor([[5, 6], 
                      [7, 8]])
    res = c * d
    print("二维张量点乘结果：\n", res)
    # 输出：
    # tensor([[ 5, 12],
    #         [21, 32]])


def matmul_demo():
    '''    
    矩阵乘法（Matrix Multiplication）：用于两个矩阵的乘法运算，要求第一个矩阵的列数等于第二个矩阵的行数。

    '''
    # 示例1：一维张量点积（最标准场景）
    a = torch.tensor([1, 2, 3])
    b = torch.tensor([4, 5, 6])
    # 方式1：torch.dot()（仅支持一维）
    res = a@b
    print("一维张量点积结果（torch.dot()）：", res)  # 输出：32

    # 示例2：matmul 实现矩阵乘法，等价于二维张量的“点积扩展                    
    c = torch.tensor([[1, 2], 
                        [3, 4]])  # shape (2,2)
    
    d = torch.tensor([[5, 6], 
                        [7, 8]])  # shape (2,2)   
                        
    #@ 是 matmul 的运算符形式                                      
    res = (c @ d)        
    
    #实现矩阵乘法,c的行对位d的列元素相乘，再求和  
    #    c[0][0]=1*5 + 2*7=19   
    #    c[0][1]=1*6 + 2*8=22
    #    c[1][0]=3*5 + 4*7=43
    #    c[1][1]=3*6 + 4*8=50

    print("二维张量矩阵乘法结果（torch.matmul()）：\n", res)
    # 输出：
    # tensor([[19, 22],
    #         [43, 50]])



def create_cnn_demo():
    # 文件路径
    file_path=os.path.dirname(__file__)+"/data/liudehua.jpg"

    # 读取图像:640*640*3
    img=plt.imread(file_path)

    # 显示图像
    # plt.imshow(img)
    # plt.axis("off")
    # plt.title("Liudehua")
    # plt.show()

    # 构建卷积层
    conv_layer = torch.nn.Conv2d(in_channels=3,         # 输入通道数，RGB图像为3通道
                                 out_channels=10,       # 输出通道数，即卷积核的数量
                                 kernel_size=(5,5),     # 卷积核大小，3x3    
                                 stride=2,              # 卷积核在输入图像上滑动的步长，默认值为1
                                 padding=1              # 填充大小，默认值为0
                                 )
    # 转为tensor张量
    img_tensor = torch.tensor(img)
    
    # 图像维度重排:640*640*3 -> 3*640*640
    img_tensor=img_tensor.permute((2,0,1))

    # 图像维度扩展:3*640*640 -> 1*3*640*640
    img_tensor = img_tensor.unsqueeze(0)   #在第0维添加一个维度，变为1*3*640*640，以符合卷积层的输入要求

    # 将图像送进卷积层,先转为torch.float32浮点数类型
    output_layer = conv_layer(img_tensor.to(torch.float32))
    print("卷积层输出特征图形状：", output_layer.shape)  # 输出：torch.Size([1, 10, 640, 640])

    # 显示特征图，每个通道单独显示
    plt.figure(figsize=(10, 8))
    for i in range(10):
        plt.subplot(2, 5, i+1)
        plt.imshow(output_layer[0, i].detach().numpy(), cmap="gray")
        plt.axis("off")
        plt.title(f"filter {i+1},卷积后特征图",size=8)
    plt.show()

if __name__=="__main__":
    #mul_demo()
    #matmul_demo()
    create_cnn_demo()