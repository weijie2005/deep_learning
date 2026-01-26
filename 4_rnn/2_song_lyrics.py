import torch
import torch.nn as nn
import os
import jieba
from torch.utils.data import DataLoader


'''
歌词生成模型实例，通过学习周杰伦的歌词，生成新的歌词。

'''


def get_lyrics_data(filename):
    '''
    从文件中读取歌词数据，返回一个列表，每个元素为一首歌词。
    '''    
    with open(filename, 'r', encoding='utf-8') as f:
        lyrics_data = f.readlines()

    return lyrics_data


def lyrics_vectorize(lyrics_data):
    '''
    对歌词数据进行分词并转成向量，返回一个列表，每个元素为一首歌词的向量表示。
    
    '''

    #jeiba分词,并将所有歌词连接成一个字符串，all_words是有序的，RNN模型需要有序的输入
    all_words=jieba.lcut(''.join(lyrics_data))

    # 创建词汇表：1.去重,用于存储唯一的歌词单词，是无重复的。
    unique_words = []    
    for word in all_words:
        if word not in unique_words:
            unique_words.append(word)

    # 创建词汇表：2.将去重后的单词映射到一个唯一的整数索引
    word_to_index={}    
    for i,word in enumerate(unique_words):
        word_to_index[word]=i        
    #print(word_to_index)

    # 通过词汇表将all_words语料转换为索引序列，即将语料都转成了整数索引值corpus_ids
    corpus_ids = []
    for word in all_words:
        corpus_ids.append(word_to_index[word])

    # 打印前100个索引值对应的单词，验证转换是否正确
    # for i,w in enumerate(corpus_ids):
    #     if i<=100:
    #         print(i,w,unique_words[w])
    
    # 词汇表大小
    word_count=len(unique_words)

    return corpus_ids, word_to_index, unique_words,word_count


class LyricsDataset(torch.utils.data.Dataset):
    '''    
    自定义歌词数据集，用于将歌词数据转换为模型可训练的数据集。
    特征与目标值：
        1. 输入序列： 长度为seq_len的整数索引序列，用于表示输入的歌词。
        2. 目标序列： 长度为seq_len的整数索引序列，每个位置上的元素是，输入序列对应位置的下一个单词，即预测的词就是输入序列对应位置的下一个单词。
    '''
    def __init__(self, corpus_ids, seq_len):
        super(LyricsDataset, self).__init__()
        self.corpus_ids = corpus_ids            # 歌词数据的整数索引序列
        self.seq_len = seq_len                  # 序列长度，即每个样本的输入和目标序列的长度
        self.word_count=len(self.corpus_ids)    # 语料库中的总单词数

    def __len__(self):
        return len(self.corpus_ids)//self.seq_len   # 数据集的样本数，每个样本包含seq_len个单词

    def __getitem__(self, idx):
        start=min(max(idx,0),self.word_count-self.seq_len-2)    # 确保索引在有效范围内
        x=self.corpus_ids[start:start+self.seq_len]             # 输入序列，长度为seq_len
        y=self.corpus_ids[start+1:start+1+self.seq_len]         # 目标序列，长度为seq_len,每个位置上的元素是，输入序列对应位置的下一个单词
        return torch.tensor(x),torch.tensor(y)


class LyricsGenerator(nn.Module):
    '''
    歌词生成模型，基于RNN模型。
        1.嵌入层： 将整数索引值转换为稠密向量表示，每个索引值对应一个固定维度的向量。
        2.RNN层： 接收嵌入向量序列作为输入，输出隐藏状态序列。
        3.全连接层： 将RNN层的输出映射到词汇表大小，每个位置上的输出对应一个单词的概率分布。
    '''
    def __init__(self, word_count, embedding_dim=64, hidden_dim=128, num_layers=2):
        super(LyricsGenerator, self).__init__()
        #1. 嵌入层： 将整数索引值转换为稠密向量表示，每个索引值对应一个固定维度的向量。
        self.embedding_layer = nn.Embedding(num_embeddings=word_count,          # 词汇表大小，即唯一单词的数量
                                      embedding_dim=embedding_dim)              # 每个单词的向量维度64

        #2. RNN层： 接收嵌入向量序列作为输入，输出隐藏状态序列。
        self.rnn_layer = nn.RNN(input_size=embedding_dim,                   # 输入维度，即嵌入向量的维度64
                                      hidden_size=hidden_dim,               # 隐藏状态维度128
                                      num_layers=num_layers,                # 隐藏层数量2
                                     )             

        #3. 全连接层： 将RNN层的输出映射到词汇表大小，每个位置上的输出对应一个单词的概率分布。
        self.fc_layer = nn.Linear(in_features=hidden_dim,       # 隐藏状态维度128
                            out_features=word_count)            # 词汇表大小，即唯一单词的数量

    def forward(self, input,hidden):
        #1. 嵌入层:将输入序列转换为嵌入向量序列
        embedding = self.embedding_layer(input)

        #2. RNN层： 接收嵌入向量序列作为输入，输出隐藏状态序列,
        # 隐藏状态形状：(num_layers, batch_size, hidden_dim)
        rnn_output,hidden = self.rnn_layer(embedding.transpose(0,1),hidden)

        #3. 全连接层： 将RNN层的输出映射到词汇表大小，每个位置上的输出对应一个单词的概率分布。
        output = self.fc_layer(rnn_output.reshape(-1, hidden_dim))
      

        return output,hidden
    
    def init_hidden(self, batch_size):
        '''
        初始化隐藏状态，返回一个全零张量，形状为(num_layers, batch_size, hidden_dim)。
        '''
        return torch.zeros(self.rnn_layer.num_layers, batch_size, self.rnn_layer.hidden_size)

def train_model(model, dataset, num_epochs,learning_rate,batch_size,model_name):
    '''
    功能： 训练模型，返回训练好的模型。
    参数：
        model (LyricsGenerator): 歌词生成模型。
        dataset (LyricsDataset): 歌词数据集。
        num_epochs (int): 训练轮数。
        learning_rate (float): 学习率。
        batch_size (int): 批次大小。
        model_name (str): 模型保存路径。
        
    '''
    
    #1.定义损失函数
    criterion=nn.CrossEntropyLoss()

    #2.定义优化器
    optimizer=torch.optim.Adam(model.parameters(),lr=learning_rate)

    #3.批次训练
    for epech in range(num_epochs):        
        #loader数据
        dataset_loader=DataLoader(dataset,batch_size=batch_size,shuffle=True)

        loss_sum=0
        loss_count=0.01

        # 遍历数据集
        for x,target in dataset_loader:
            # 初始化隐藏状态
            # 根据实际批次大小初始化隐藏状态，而不是使用固定的batch_size
            actual_batch_size = x.size(0)  # 获取当前批次的实际大小
            h0 = model.init_hidden(actual_batch_size)

            # 数据送到模型：将输入序列和隐藏状态送到模型中，得到输出序列和新的隐藏状态
            y_predict,hidden=model(x,h0)

            # torch要求换位[batch_size, seq_len]->[seq_len, batch_size]，同时目标序列转换为一维向量，用于计算损失
            y=torch.transpose(target,0,1).reshape(-1)

            # 计算损失
            loss=criterion(y_predict,y)

            # 累加损失
            loss_sum+=loss.item()
            # 累加样本数
            loss_count+=1
            
            # 梯度清零
            optimizer.zero_grad()
            # 反向传播
            loss.backward()
            # 更新参数
            optimizer.step()

        # 打印损失
        print(f"Epoch [{epech+1}/{num_epochs}], Loss: {loss_sum/loss_count:.4f}")

    #4. 保存模型
    torch.save(model.state_dict(), model_name)
    

def predict_lyrics(model, start_word, unique_words, word_to_index,max_len):
    '''
    预测歌词，返回生成的歌词。
    参数：
        model (LyricsGenerator): 歌词生成模型。
        start_word (str): 开始的单词。
        max_len (int): 最大预测长度。
        unique_words (list): 词汇表，包含所有唯一的单词。
        word_to_index (dict): 单词到索引的映射，用于将单词转换为索引值。
       
    '''
    # 初始化隐藏状态 - 预测时批次大小为1
    prediction_batch_size = 1
    h0 = model.init_hidden(prediction_batch_size)

    # 将开始单词转换为索引值
    word_index = word_to_index[start_word]

    # 存储生成的歌词
    lyrics_index = []
    lyrics_word=[]

    for _ in range(max_len):
        # 数据送到模型：将输入序列和隐藏状态送到模型中，得到输出序列和新的隐藏状态
        y_predict,hidden=model(torch.tensor([[word_index]]),h0)

        # 取输出序列的最后一个位置的预测结果
        y_predict=y_predict[-1,:]

        # 取概率最大的索引作为预测结果
        word_index=torch.argmax(y_predict).item()

        # 追加到生成的歌词中
        lyrics_index.append(word_index)
        
        # 更新隐藏状态
        #h0 = hidden

    for idx in range(len(lyrics_index)):
        # 将索引转换为单词
        predicted_word=unique_words[idx]
        lyrics_word.append(predicted_word)

    # 合并生成的歌词
    return ' '.join(lyrics_word)

if __name__ == '__main__':
    # 读取歌词数据
    current_directory = os.path.dirname(os.path.abspath(__file__))
    filename=os.path.join(current_directory, "data/jaychou_lyrics.txt")
    model_name=os.path.join(current_directory, "model/lyrics_model.pth")

    # 读取歌词数据
    lyrics_data = get_lyrics_data(filename)
    
    # 超参数
    batch_size = 32
    learning_rate = 0.001
    num_epochs = 10

    embedding_dim = 64
    hidden_dim = 128
    num_layers = 2
    seq_len = 5


    # 对歌词数据进行分词并转成索引值
    corpus_ids, word_to_index, unique_words,word_count = lyrics_vectorize(lyrics_data)

    # 创建歌词数据集：组合生成特征和目标值
    dataset = LyricsDataset(corpus_ids, seq_len)
    
    # 训练模型  self, word_count, embedding_dim=64, hidden_dim=128, num_layers=2  
    model=LyricsGenerator(word_count,           # 词汇表大小
                            embedding_dim,        # 嵌入向量维度64
                            hidden_dim,           # 隐藏状态维度128                             
                            num_layers)           # 隐藏层数量2
        
    # 如果模型存在就预测，不存在则训练
    if os.path.exists(model_name):
        # 加载模型参数
        model.load_state_dict(torch.load(model_name))

        start_word = '我'
        # 预测歌词
        new_lyrics = predict_lyrics(model,          # 模型
                                    start_word,     # 开始的单词
                                    unique_words,   # 词汇表
                                    word_to_index,  # 单词到索引的映射
                                    max_len=100)    # 最大预测长度100

        # 打印预测的歌词
        print(new_lyrics)
    else:    
        train_model(model, dataset, num_epochs,learning_rate,batch_size,model_name)


