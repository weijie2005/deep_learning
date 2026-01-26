import torch
import torch.nn as nn
import jieba


'''
一、循环神经网络（Recurrent Neural Network, RNN）
    1.基本特性：
        1. 循环神经网络（RNN）：是一种特殊的神经网络，用于处理有序列特点的数据。    
        2. 它的核心思想是在每个时间步上，将当前输入和上一个时间步的隐藏状态结合起来，生成当前时间步的隐藏状态。
        3. 循环神经网络：可以用于处理文本语句、语音、天气、金融等与时间顺序等序列有关的数据。
        4. 文本数据就是有序列特性的，如果顺序不同，表达的意思也就不一样。 
        
    2.RNN的计算方法：
        1. 初始化隐藏状态：h_0 = 0
        2. 遍历输入序列：
            2.1 计算当前时间步的隐藏状态：h_t = tanh(W_hh * h_(t-1) + W_xh * x_t)
            2.2 计算当前时间步的输出：y_t = softmax(W_hy * h_t)
        3. 返回输出序列和最终隐藏状态

        4. 使用激活函数：
            4.1 隐藏状态激活函数：tanh
            4.2 输出激活函数：softmax

        5. 两个输入即：
            5.1 当前输入序列：x = [x_1, x_2, ..., x_T]
            5.2 上一个隐藏状态序列：h = [h_0, h_1, ..., h_T]

        6. 两个输出即：
            6.1 输出序列：y = [y_1, y_2, ..., y_T]
            6.2 当前隐藏状态：h_T
            6.3 其实输出和隐藏状态是相同的值，都是加权和与激活函数的输出，即y1=h1。
            6.4 输出h和y其中h=y,h用于下次循环作为上一个时间步的隐藏状态,y输出到输出层，所以再用新的h作为下一个时间步态的隐藏状态加上下一个x作为输入，如此循环。

        7. 计算公式:
            Y_t=H_t = tanh(W_hh * H_(t-1) + W_xh * X_t)
            其中：
            H_t：当前时间步的隐藏状态
            Y_t：当前时间步的输出
            H_(t-1)：上一个时间步的隐藏状态
            W_hh：隐藏状态权重矩阵
            W_xh：输入权重矩阵
            X_t：当前时间步的输入
            
            
    3.RNN的循环结构：
        1. 循环神经网络：是由一个神经元组成的循环结构，通过循环将h,x输入到神经元中，计算当前时间步的隐藏状态h_t和输出y_t。
        2. 每个时间步上，循环神经网络将当前输入和上一个时间步的隐藏状态结合起来，生成当前时间步的隐藏状态。
        3. 循环神经网络的计算可以用循环结构来表示，即每个时间步的计算都依赖于前一个时间步的隐藏状态。
        4. 循环神经网络的参数包括：输入权重矩阵W_xh、隐藏状态权重矩阵W_hh、输出权重矩阵W_hy、偏置项b_h、b_y。

    4. 循环神经网络的训练：
        4.1 损失函数：交叉熵损失函数
        4.2 优化算法：随机梯度下降（SGD）或其 variants（如 Adam）

二、RNN相关API
    1.实例化RNN
        RNN=torch.nn.RNN(input_size=embedding_dim,    # 输入维度，即词向量的维度
                hidden_size=hidden_size,     # 隐藏状态h维度,即当前神经元的输出维度
                num_layers=num_layers,       # 循环层数,默认1层
                batch_first=True)            # 输入输出张量的第一个维度是否为批量大小
        
    2.调用
        output,hidden=RNN(x,h0)
        # 其中output是所有时间步的输出，hidden是最后一个时间步的隐藏状态

        # 以下是在torch中RNN的输入输出形状：transformer是不同的形状，需调整顺序。
        1.输入数据
            # x输入形状：(seq_len,batch_size, input_size)，即[句子长度，批量大小，词向量维度]
            # h0输入形状：(num_layers, batch_size, hidden_size)，即[循环层数，批量大小，隐藏状态维度]
        2.输出数据
            # output输出形状：(seq_len, batch_size, hidden_size),即[句子长度，批量大小，隐藏状态维度]
            # hn隐藏状态形状：与h0一样(num_layers, batch_size, hidden_size)，即[循环层数，批量大小，隐藏状态维度]

        3.其中seq_len：句子长度设成一样大小，不可以不同，否则会报错。
            # 例如：句子1："我 是 一个 学生"，句子2："我 来自 北京"，如果设成一样大小，即4个句子长度，那么句子1就会被填充为"我 是 一个 学生"，句子2就会被填充为"我 来自 北京 0"。
            # 其中0是填充符，用于填充短句子，使所有句子长度相同。

            

三、NLP自然语言数据预处理：
    1.分词：
        1.1 什么是分词？
        1.2 分词的作用:
        1.3 常用的分词工具:Jieba

    2.嵌入层-Embedding()向量化：
        1.将文本转换为数值索引序列
        2.使用嵌入层将索引序列转换为向量矩阵
        3.将向量矩阵输入到RNN模型中进行训练



'''

'''
NLP自然语言处理之数据预处理实例
    1.分词
    2.嵌入层-Embedding()向量化
'''


def embedding_layer(text,embedding_dim):
    '''
        定义嵌入层
    参数：
        embedding_dim (int): 嵌入向量维度
    返回：
        嵌入层
    '''
    #2. 对文本进行分词
    seg_list=jieba.lcut(text)
    print("分词结果:",seg_list)

    #3. 使用集合去重，并构建词汇表
    seg_list=list(set(seg_list))
    print("去重后的分词结果:",seg_list)

    # 定义嵌入层
    embedding_layer=nn.Embedding(num_embeddings=len(seg_list),  # 词汇表大小
                                 embedding_dim=embedding_dim)   # 嵌入向量维度

    for i,word in enumerate(seg_list):
        vec=embedding_layer(torch.tensor(i))    # 获取索引i对应的嵌入向量,i要转为tensor类型
        # 打印词汇表中的每个单词和对应的索引、向量
        print(f"单词: {word}  索引: {i}  向量: {vec}")

    return embedding_layer


def rnn_demo():
    # 定义RNN模型
    rnn=nn.RNN(input_size=128,        # 输入维度，即词向量的维度
                hidden_size=64,     # 隐藏状态h维度,即当前神经元的输出维度
                num_layers=10,       # 循环层数,默认1层
                batch_first=True)            # 输入输出张量的第一个维度是否为批量大小

    # x输入形状：(seq_len,batch_size, input_size)，即[句子长度，批量大小，词向量维度]
    # h0输入形状：(num_layers, seq_len, hidden_size)，即[循环层数，批量大小，隐藏状态维度]
    x=torch.randn(12,24,128)       # 输入数据，句子长度12个，batch_size=24，每个时间步128维向量
    h0=torch.zeros(10,12,64)      # 初始隐藏状态，10层循环，seq_len=12，每个样本64维隐藏状态

    # 调用RNN模型
    output,hidden=rnn(x,h0)
    print("output:",output.shape)    # 输出形状：(seq_len, batch_size, hidden_size)，即[句子长度，批量大小，隐藏状态维度]
    print("hidden:",hidden.shape)    # 隐藏状态形状：与h0一样(num_layers, batch_size, hidden_size)，即[循环层数，批量大小，隐藏状态维度]


if __name__ == '__main__':
    #1. 定义嵌入向量维度    
    embedding_dim=4

    #2. 定义文本数据
    text="北京冬奥的进度条已经过半，不少外国运动员在完成自已的比赛后踏上归途。"

    #3. 调用嵌入层函数
    #embedding_layer(text,embedding_dim)

    #4. 调用RNN模型
    rnn_demo()


    
