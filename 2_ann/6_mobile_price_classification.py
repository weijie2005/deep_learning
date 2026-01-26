import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import os
from torchsummary import summary
'''

二手手机价格分类实例：通过神经网络模型对二手手机的价格进行分类和预测。

'''

def preprocess_data(file_name):
    '''
    对数据进行预处理，包括合并数据集、划分训练集和测试集、标准化特征等。
    
    参数:
    file_name (str): 包含特征和价格范围的csv文件名。
    
    返回:
    train_dataset, test_dataset: 包含训练集和测试集特征和标签的TensorDataset对象。
    '''
    # 获取当前脚本所在路径
    file_path = os.path.abspath(__file__)
    current_dir=os.path.dirname(file_path)
    # 加载数据集
    data = pd.read_csv(os.path.join(current_dir,file_name))
    
    # 查看数据由以下结果可知，
    # print("数据形状:", data.shape)
    # print("数据列名:", data.columns.tolist())

     # 因为报错，所以检查目标值price_range列的值分布，确保没有异常值
    print(f"数据集中price_range的唯一值: {sorted(data['price_range'].unique())}")

    # 提取特征和标签，去掉price_range列，维度为20
    x=data.drop('price_range', axis=1)
    y=data['price_range']

    # 划分数据集为训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # 将数据转换成tensor格式,并将标签转换为float32类型,以便torch创建模型使用
    X_train = torch.tensor(X_train.values, dtype=torch.float32)
    X_test = torch.tensor(X_test.values, dtype=torch.float32)

    y_train = torch.tensor(y_train.values, dtype=torch.int64)
    y_test = torch.tensor(y_test.values, dtype=torch.int64)

    # 使用tensorDataset封装训练集和测试集
    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    test_dataset = torch.utils.data.TensorDataset(X_test, y_test)
    
    return train_dataset, test_dataset


# 定义神经网络模型
class Mymodel(torch.nn.Module):
    '''
    定义神经网络模型
    参数:
    input_dim (int): 输入特征维度，即每个样本的特征数量。
    output_dim (int): 输出特征维度，即分类的类别数量。
    '''

    def __init__(self, input_dim, output_dim):
        #1. 调用父类的初始化方法
        super(Mymodel,self).__init__()        

        #2. 创建第一个隐藏层，输入特征20，输出128
        self.layer1 = torch.nn.Linear(in_features=input_dim, out_features=128)
        

        #3. 创建第二个隐藏层，输入特征128，输出256
        self.layer2 = torch.nn.Linear(in_features=128, out_features=256)
        
        
        #4. 创建第三个隐藏层，输入特征256，输出512
        self.layer3 = torch.nn.Linear(in_features=256, out_features=1024)


        #6. 创建输出层，输入特征256，输出特征4
        self.output_layer = torch.nn.Linear(in_features=1024, out_features=output_dim)
        

        #7. 创建失活层，以防过拟合
        self.dropout = torch.nn.Dropout(p=0.2)
    
    def forward(self, x):
        # 隐藏层1
        x = self.layer1(x)
        x = self.dropout(x) # 应用dropout层,失活后再使用ReLU激活函数
        x = torch.nn.functional.relu(x)            # 隐藏层1使用ReLU激活函数       

        # 隐藏层2
        x = self.layer2(x)
        x = self.dropout(x) # 应用dropout层，失活后再使用ReLU激活函数        
        x = torch.nn.functional.relu(x)            # 隐藏层2使用ReLU激活函数

        # 隐藏层3
        x = self.layer3(x)
        x = self.dropout(x) # 应用dropout层，失活后再使用ReLU激活函数        
        x = torch.nn.functional.relu(x)            # 隐藏层3使用ReLU激活函数           

        # 输出层
        x = self.output_layer(x)

        # 输出层使用softmax函数，将输出转换为概率分布。由于在后面使用了交叉熵损失函数计算损失时，已经包含了softmax函数，所以这里不需要再使用softmax函数。
        #x = self.softmax(x)           

        return x



# 模型训练
def train(model,train_dataset, input_dim, output_dim, learning_rate, batch_size, num_epochs,model_path):
    '''
    训练模型
    
    参数:
    train_dataset (torch.utils.data.TensorDataset): 包含训练集特征和标签的TensorDataset对象。
    input_dim (int): 输入特征维度，即每个样本的特征数量。
    output_dim (int): 输出特征维度，即分类的类别数量。
    learning_rate (float): 学习率。
    batch_size (int): 批次大小。
    num_epochs (int): 训练轮数。
    
    返回:
    model: 训练好的模型。
    '''

    # 定义损失函数:交叉熵损失函数
    criterion = torch.nn.CrossEntropyLoss()

    #定义优化器：随机梯度下降优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # 训练模型
    for epoch in range(num_epochs):
        # 每轮次加载训练数据
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        total_loss = 0.0
        total_num = 0

        for inputs, targets in train_loader:
            #梯度清零
            optimizer.zero_grad()

            #将输入数据传递给模型,获得输出预测结果
            y_predict = model(inputs)

            # 计算损失，交叉熵损失函数计算真实值和预测值之间的差异
            loss = criterion(y_predict, targets)

            # 反向传播
            loss.backward()

            # 更新参数
            optimizer.step()

            # 累计损失和样本数量
            total_loss += loss.item() 
            total_num += 1
        
        # 打印每个epoch的损失：如果出现nan说明出现了梯度爆炸，需要调整学习率，如果出现损失loss不下降说明梯度消失了，需要调整模型结构
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss / total_num:.4f}")

    # 保存模型
    torch.save(model.state_dict(), model_path)  
    print(f"Model saved to {model_path}")

    return model

def predict_model(model, test_dataset, batch_size):
    '''
    模型预测
    
    参数:
    model: 训练好的模型。
    test_dataset: 测试数据集。
    
    返回:
    accuracy: 模型在测试集上的准确率。
    '''

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)        

    #遍历测试集，计算准确率
    correct = 0
    for x, y_target in test_loader:
        #将输入数据传递给模型,获得输出预测结果,y_predict是一个4维张量，每个样本有4个输出，每个输出对应一个类别的概率
        y_predict = model(x)

        # 计算预测结果中概率最大的类别索引,即[0,1,2,3]
        y_index = torch.argmax(y_predict, dim=1)

        # 累计正确预测的样本数量,
        # 这里使用sum()函数计算预测正确的样本数量，因为(predicted == y_target)返回的是一个布尔张量，True表示预测正确，False表示预测错误。
        # 调用sum()函数可以统计True的数量，即预测正确的样本数量。
        correct += (y_index == y_target).sum()

    accuracy = 100 * correct / len(test_dataset)
    print(f"Test Accuracy: {accuracy:.2f}%")
    return accuracy


if __name__ == '__main__':
    # 定义数据集文件和模型保存路径    
    file_name = 'data/mobile_prices.csv'   
    model_path = os.path.join(os.path.dirname(__file__), 'model/mobile_price_model.pth')

    # 定义超参数
    input_dim = 20      # 输入特征维度，即每个样本的特征数量
    output_dim = 4      # 输出特征维度，即分类的类别数量
    learning_rate = 0.0001   # 学习率
    batch_size = 32         # 批次大小
    num_epochs = 1000        # 训练轮数

    # 加载数据集
    train_dataset, test_dataset = preprocess_data(file_name)

    # 实例化模型
    model = Mymodel(input_dim, output_dim)

    # 如果模型存在，则使用模型预测，不存在就训练模型
    if os.path.exists(model_path):        
        model.load_state_dict(torch.load(model_path,weights_only=True))
        predict_model(model, test_dataset, batch_size)
    else:
        # 开始训练模型
        train(model,train_dataset, input_dim, output_dim, learning_rate, batch_size, num_epochs,model_path)    


   
