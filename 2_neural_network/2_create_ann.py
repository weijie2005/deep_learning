import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchsummary import summary

'''

5. 选择激活函数：
    1. 二分类问题：sigmoid 函数,指用在输出层中
    2. 多分类问题：softmax 函数,指用在输出层中
    3. 隐藏层：ReLU 函数或 tanh 函数
    4. 输出层：sigmoid 激活函数，多分类问题使用softmax函数
    5. 回归类激活函数：无，不需要，直接输入即输出
    6. 指数激活函数：sigmoid, tanh
    7. 分段激活函数：ReLU, [ReLU增加超参后的变形：leaky ReLU ，PReLU，RReLU,ELU]
    8. 隐层主要使用ReLU函数，当期效果不好，造成神经元死亡问题时，使用leaky ReLU函数。
    9. 回归问题使用identity函数，即不使用激活函数，直接输出。

一、神经网络初始化参数：
    1.w权重参数：
        1.平均分布：w参数从[-1/sqrt(n), 1/sqrt(n)]之间随机初始化，其中n是上一层的神经元数量。
        2.正态分布：w参数从均值为0，标准差为1/sqrt(n)的正态分布中随机初始化。
        3.HE(kaiming)初始化：
            1. 平均分布HE初始化：w参数是[-sqrt(6/fan_in), sqrt(6/fan_in)]平均初始化，其中fan_in是输入神经元数量。
            2. 正态分布HE初始化：w参数是[sqrt(2/fan_in)]的正态分布初始化，其中fan_in是输入神经元数量。

        4.xavier初始化：
            1. 平均分布xavier初始化：w参数是[-sqrt(6/fan_in+fan_out), sqrt(6/fan_in+fan_out)]平均初始化，其中fan_in是输入神经元数量，fan_out是输出神经元数量。
            2. 正态分布xavier初始化：w参数是[sqrt(2/fan_in+fan_out)]的正态分布初始化，其中fan_in是输入神经元数量，fan_out是输出神经元数量。

    2.b偏置参数：
        1.全0初始化：b参数初始化为0。
        2.全1初始化：b参数初始化为1。
        3.固定值初始化：b参数初始化为一个固定的标量值，如0.1。


二、神经网络的创建过程：
    1. 定义神经网络的层：
        1. 输入层：根据输入数据的维度定义输入层的神经元数量。
        2. 隐藏层：根据任务需求和经验定义隐藏层的神经元数量和层数，每个隐藏层使用ReLU激活函数。
        3. 输出层：根据输出数据的维度定义输出层的神经元数量，根据任务需求选择合适的激活函数。
    2. 初始化参数：
        1. 对每个层的权重参数w进行初始化，这里选择xavier初始化。
        2. 对每个层的偏置参数b进行初始化，这里选择全0初始化。
    3. 定义前向传播：
        1. 对输入数据进行前向传播，通过每个层的权重参数和偏置参数进行计算，得到输出结果。
    4. 定义损失函数：
        1. 根据任务需求选择合适的损失函数，如均方误差损失、交叉熵损失等。
    5. 定义优化器：
        1. 选择合适的优化器，如随机梯度下降（SGD）、Adam等。
    


'''

'''
    创建一个神经网络实例，包含一个输入层，一个隐藏层，一个输出层
        1. torchsummary工具包用于打印神经网络的参数信息

'''

class MYANN_MODEL(nn.Module):
    def __init__(self):
        #1. 调用父类的初始化方法
        super(MYANN_MODEL, self).__init__()

        #2. 创建第一个隐藏层，输入3特征，输出3特征
        self.layer1 = nn.Linear(in_features=3, out_features=3)

        #2.1. 初始化第一层的w权重参数，b偏置参数为0
        nn.init.kaiming_normal_(self.layer1.weight)
        nn.init.zeros_(self.layer1.bias)        

        #3. 创建第二个隐藏层，输入3特征，输出2特征
        self.layer2 = nn.Linear(in_features=3, out_features=2)

        #3.1. 初始化第二层的w权重参数，b偏置参数为0
        nn.init.xavier_normal_(self.layer2.weight)
        nn.init.zeros_(self.layer2.bias)        

        #4. 创建输出层，输入2特征，输出2特征
        self.out=nn.Linear(2,2)

        #4.1. 初始化输出层的w权重参数，b偏置参数为0
        nn.init.uniform_(self.out.weight)
        

    def forward(self, x):
        #1. 前向传播，通过第一个隐藏层，并对第一个隐藏层的输出进行ReLU激活函数
        x_layer1 = self.layer1(x)        
        x_layer1 = torch.relu(x_layer1)

        #2. 前向传播，通过第二层，并对第二个隐藏层的输出进行LeakyReLU激活函数
        x_layer2 = self.layer2(x_layer1)
        x_layer2 = torch.tanh(x_layer2)

        #3. 传到输出层，并对输出层的输出使用softmax函数
        out_layer = self.out(x_layer2)
        out_layer = torch.softmax(out_layer, dim=1)
               
        return out_layer


if __name__ == '__main__':
    # 实例化神经网络MYANN类对象
    my_ann = MYANN_MODEL()

    # 打印神经网络的参数
    print("神经网络的参数:",my_ann.parameters())

    # 产生一个随机输入张量, 5个样本,样本数可以根据实际情况调整, 每个样本3个特征，但特征数要与创建的输入层的in_features一致
    random_input = torch.randn(5, 3)
    
    # 数据通过神经网络模型训练
    output=my_ann(random_input)
    print("经过神经网络的输出:",output)
    print("经过神经网络的输出形状:",output.shape)

    output_bias=my_ann.out.bias
    print("输出层的偏置参数:",output_bias)
    
    output_weight=my_ann.out.weight
    print("输出层的权重参数:",output_weight)

    # 打印神经网络的参数信息
    summary(my_ann, input_size=(3,),batch_size=8)

    # 遍历参数，打印参数名称和参数值
    for name, param in my_ann.named_parameters():
        print(name, param)  