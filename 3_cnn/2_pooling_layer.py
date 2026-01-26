import torch
import torch.nn as nn

'''
池化层，用于提取特征图的局部信息。
'''

def pooling_demo():
    # 定义多通道数据3*3*3
    input=torch.tensor([
        [
            [1,2,3],
            [4,5,6],
            [7,8,9]
        ],
        [
            [10,11,12],
            [13,14,15],
            [16,17,18]
        ],
        [
            [19,20,21],
            [22,23,24],
            [25,26,27]
        ]
    ],dtype=torch.float32)



    # 池化层：最大池化
    max_pooling = nn.MaxPool2d(kernel_size=(2,2),   # 池化核大小
                           stride=1,            # 步长
                           padding=0            # 填充      
                           )
    # 池化层的输出
    output = max_pooling(input)
    print("最大池化输出:",output)


    # 池化层：平均池化
    avg_pooling = nn.AvgPool2d(kernel_size=(2,2),   # 池化核大小
                           stride=1,            # 步长
                           padding=0            # 填充      
                           )
    # 池化层的输出
    output = avg_pooling(input)
    print("平均池化输出:",output)

if __name__ == '__main__':
    pooling_demo()