import torch
import torch.nn as nn
from torchvision.transforms import Compose, ToTensor, Normalize
from torchvision.datasets import CIFAR10
import os
from torchsummary import summary


'''

通过卷积层和池化层提取图片特征，再经过全连接层神经网络进行分类训练。

'''


class ImageClassifier(nn.Module):
    '''
    图片分类神经网络模型，包含两个卷积层和两个全连接层。
    '''
    def __init__(self):
        # 初始化父类
        super(ImageClassifier,self).__init__()

        # 第一个卷积层
        self.conv1=nn.Conv2d(in_channels=3,     # 输入通道数，RGB图片为3通道
                             out_channels=6,    # 输出通道数，即卷积核的数量
                             kernel_size=[3,3],     # 卷积核大小
                             stride=1,          # 步长，默认值为1
                             padding=0)         # 填充大小，保持输入输出大小相同
        # 第一个池化层
        self.pool1=nn.MaxPool2d(kernel_size=[2,2],   # 池化核大小
                               stride=2,            # 步长，默认值为1
                               padding=0)           # 填充大小

        # 第二个卷积层
        self.conv2=nn.Conv2d(in_channels=6,out_channels=16,kernel_size=3,stride=1,padding=0)

        # 第二个池化层
        self.pool2=nn.MaxPool2d(kernel_size=[2,2], stride=2, padding=0)   
        
        # 第一个全连接层
        self.fc1=nn.Linear(in_features=576,out_features=256)

        # 第二个全连接层
        self.fc2=nn.Linear(in_features=256,out_features=64)

        # 输出层
        self.output=nn.Linear(in_features=64,out_features=10)
    
    def forward(self,x):
        # 第一个卷积层并激活
        x=torch.relu(self.conv1(x))
        
        # 第一个池化层
        x=self.pool1(x)
        
        # 第二个卷积层并激活
        x=torch.relu(self.conv2(x))

        # 第二个池化层
        x=self.pool2(x)

        # 展平张量,将卷积层输出的特征图展平为一维向量,使用torch.reshape函数
        x=torch.reshape(x,[x.size(0),-1])

        # 第一个全连接层并激活
        x=torch.relu(self.fc1(x))

        # 第二个全连接层
        x=torch.relu(self.fc2(x))

        # 输出层
        output=self.output(x)

        return output


def get_data(save_path):
    '''
        下载CIFAR10数据集 cifar-10-batches-py
    '''
    # 如果存在数据集文件，直接加载数据集
    if os.path.exists(f'{save_path}/cifar-10-batches-py'):
        train_data=CIFAR10(root=f'{save_path}',train=True,transform=Compose([ToTensor()]),download=False)
        test_data=CIFAR10(root=f'{save_path}',train=False,transform=Compose([ToTensor()]),download=False)
    else:
        # 如果数据集不存在，下载数据集
        train_data=CIFAR10(root=f'{save_path}',train=True,transform=Compose([ToTensor()]),download=True)
        test_data=CIFAR10(root=f'{save_path}',train=False,transform=Compose([ToTensor()]),download=True)
    return train_data,test_data


def train_model(epochs,train_data,learning_rate,batch_size,model_save_path):
    '''    
        训练图片分类神经网络模型。

    参数：
        epochs (int): 训练轮数
        train_loader (DataLoader): 训练集数据加载器
        learning_rate (float): 学习率
        batch_size (int): 批量大小
        model_save_path (str): 模型保存路径

    返回：
        训练模型并返回模型。
    '''


    # 实例化模型
    model=ImageClassifier()
    
    # 定义损失函数: 交叉熵损失函数
    criterion=nn.CrossEntropyLoss()

    # 定义优化器: Adam优化器
    optimizer=torch.optim.Adam(model.parameters(),lr=learning_rate,betas=(0.9,0.999))

    # 定义训练循环
    for epoch in range(epochs):
        # 初始化训练损失和准确率
        train_loss=0.0
        train_sum=0.0

        #使用dataloader加载训练集
        train_loader=torch.utils.data.DataLoader(train_data,batch_size=batch_size,shuffle=True)

        # 遍历训练集
        for images,targer in train_loader:
            # 输入数据传入模型
            y_predict=model(images)

            # 计算损失: 交叉熵损失函数计算损失
            loss=criterion(y_predict,targer)

            # 清空之前的梯度
            optimizer.zero_grad()

            # 反向传播
            loss.backward()

            # 更新参数
            optimizer.step()

            # 统计训练损失
            train_loss+=loss.item()
            train_sum+=1

        # 计算平均训练损失
        loss_mean=train_loss/train_sum

        # 打印训练信息
        print(f'Epoch [{epoch+1}/{epochs}], avg Loss: {loss_mean:.4f}')

    # 保存模型
    torch.save(model.state_dict(),model_save_path)

    return model


def predict_model(model, test_data):
    """
    使用训练好的模型对测试数据进行预测
    
    参数:
        model: 训练好的模型
        test_data: 测试数据集
    
    返回:
        无返回值，打印预测结果和准确率
    """
    # 设置模型为评估模式
    model.eval()
    
    # 创建测试数据加载器
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=False)
    
    # 初始化预测正确的数量和总数量
    correct = 0
    total = 0
    
    # 不计算梯度，节省内存和计算资源
    with torch.no_grad():
        for images, labels in test_loader:
            # 将数据传入模型进行预测
            outputs = model(images)
            
            # 获取预测结果（概率最大的类别）
            _, predicted = torch.max(outputs.data, 1)
            
            # 累计总样本数
            total += labels.size(0)
            
            # 累计预测正确的样本数
            correct += (predicted == labels).sum().item()
    
    # 计算准确率
    accuracy = 100 * correct / total
    
    print(f'测试准确率: {accuracy:.2f}% ({correct}/{total})')
    
    # 可以选择性地返回准确率
    return accuracy


if __name__ == '__main__':
    # 获取当前脚本的绝对路径
    current_path=os.path.abspath(__file__)
    # 获取当前脚本所在目录
    current_dir=os.path.dirname(current_path)
    
    # 定义数据集保存路径
    dataset_save_path=f'{current_dir}/data'

    # 定义模型保存路径
    model_save_path=f'{current_dir}/model/image_model.pth'    

    # 自动判断数据集文件是否存在：如果数据集不存在，下载数据集，否则直接加载数据集
    train_data,test_data=get_data(dataset_save_path)        
    
    # 超参数
    batch_size=64
    learning_rate=0.001
    epochs=10

    # 如果模型存在则预测，否则训练模型
    if os.path.exists(model_save_path):
        # 实例化模型
        model=ImageClassifier()
        # 加载模型参数
        model.load_state_dict(torch.load(model_save_path))
        # 模型预测        
        predict_model(model,test_data)
    else:
        model=train_model(epochs,train_data,learning_rate,batch_size,model_save_path)

        # 打印模型摘要
        summary(model,input_size=(3,32,32),batch_size=batch_size)


