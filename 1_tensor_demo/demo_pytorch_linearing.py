from sklearn.datasets import make_regression
import matplotlib.pyplot as plt
import torch
from torch.utils.data import TensorDataset, DataLoader
import os
'''
演示代码，pytorch 实现线性回归：
    1.通过 make_regression 生成模拟数据
    2.将数据转换为 PyTorch 张量
    3.创建 DataLoader 用于批量训练
    4.构建模型
    5.训练模型
        1.损失函数
        2.优化器
        3.更新模型参数
        4.再循环训练
    6.评估模型
'''

def train_model(model, dataloader, criterion, optimizer, num_epochs):
    # 记录每个epoch的损失
    total_loss = []

    # 记录最佳模型的损失和状态
    best_loss = float('inf')
    best_model_state = None

    # epoch批次训练模型
    for e in range(num_epochs):
        epoch_losses = []  # 存储当前epoch所有batch的损失
        
        # 每个epoch，遍历所有样本
        for x, y in dataloader:
            # 模型预测
            y_pred = model(x.type(torch.float32))

            # 确保y是列向量，形状为[n_samples, 1],否则y原始的开状是[n_samples],与y_pred的形状[n_samples, 1]不匹配
            y=y.reshape(-1,1)   #或者扩展一个一维，y = y.type(torch.float32).unsqueeze(1)  # 将y从[n]变为[n, 1]以匹配模型输出

            # 计算损失，通过损失标准对象criterion计算：MSE
            loss = criterion(y_pred, y.type(torch.float32))
            epoch_losses.append(loss.item())  # 收集每个batch的损失
            
            # 梯度清零
            optimizer.zero_grad()

            # 反向传播，计算梯度
            loss.backward()

            # 更新模型参数
            optimizer.step()

        # 每个epoch，计算平均损失（对所有batch的损失取平均）
        epoch_loss = sum(epoch_losses) / len(epoch_losses)

        # 记录每个epoch的平均损失
        total_loss.append(epoch_loss)
        
        # 检查是否是最佳模型
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            # 保存最佳模型的状态字典
            best_model_state = model.state_dict().copy()
            
        # 打印每隔10个epoch的信息
        if (e + 1) % 10 == 0:
            print(f'Epoch [{e+1}/{num_epochs}], Loss: {epoch_loss:.4f}')
   
    # 返回损失历史和最佳模型状态
    return total_loss, best_model_state, best_loss


if __name__ == '__main__':
    # 1.通过 make_regression 生成模拟数据
    X, y, coef = make_regression(n_samples=100,   # 样本个数
                                n_features=1,     # 特征个数
                                noise=10,         # 噪声标准差
                                random_state=42,  # 随机种子，确保结果可重现
                                coef=True         # 是否返回斜率
                                )
    # 2.展示数据
    # plt.scatter(X, y, label="Data")
    # plt.plot(X, X*coef, color="red", label="True Line")
    # plt.legend()
    # plt.show()

    # 3.将数据转换为 PyTorch 张量
    X_tensor = torch.tensor(X)
    y_tensor = torch.tensor(y)

    # 4.创建 DataLoader 用于批量训练
    dataset = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(dataset,        # 数据集
                            batch_size=8,   # 每个批次的样本数
                            shuffle=True    # 打乱数据，在每个epoch开始前
                            )
    
    # 5.构建模型
    model = torch.nn.Linear(in_features=1, out_features=1)  # 输入维度1，输出维度1（即线性回归）

    # 6.定义损失函数和优化器
    criterion = torch.nn.MSELoss()  # 实例一个损失标准对象，选用均方误差损失
    optimizer = torch.optim.SGD(params=model.parameters(), lr=0.001)  # 随机梯度下降优化器

    # 7.训练模型,返回每个epoch的平均损失及最佳模型状态和最佳损失
    num_epochs = 100    # 训练100个epoch
    total_loss, best_model_state, best_loss = train_model(model, dataloader, criterion, optimizer, num_epochs)

    print("训练完成，共训练{}个epoch".format(len(total_loss)))
    print(f"初始损失: {total_loss[0]:.4f}")
    print(f"最终损失: {total_loss[-1]:.4f}")
    
    # 8.保存最佳模型
    curr_path = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(curr_path, "./models/best_linear_model.pth")
    torch.save(best_model_state, model_path)

   # 9.展示训练损失变化趋势
    plt.figure(figsize=(10, 6))
    plt.plot(range(num_epochs), total_loss, label='Training Loss')
    plt.axhline(y=best_loss, color='r', linestyle='--', label=f'Best Loss: {best_loss:.4f}')
    plt.grid(True)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss per Epoch')
    plt.legend()
    plt.show()
    
    # 10.演示如何加载保存的模型
    # print("\n演示加载保存的最佳模型:")
    # new_model = torch.nn.Linear(in_features=1, out_features=1)
    # loaded_model = torch.load(model_path)