
import torch


'''

七、神经网络的损失函数(loss function)：在文献中有其它名称如：代价函数(cost function)、目标函数(objective function)、误差函数(error function)等。
    1.什么是损失函数？
        1. 损失函数是衡量模型参数(w,b)质量好坏的函数。
        1. 损失函数是模型预测值与真实标签值之间差异的函数。
        2. 损失函数的目标是最小化模型预测结果与真实标签之间的差异，从而提高模型的预测准确性。

    2. 分类问题常用损失函数：
        1. 多分类交叉熵损失函数（Cross-Entropy Loss）：用于多分类问题，衡量模型预测的概率分布与真实标签的差异。
            L=-(y*log(y_pred)+(1-y)*log(1-y_pred))
            torch中的多分类交叉熵损失函数：torch.nn.CrossEntropyLoss()
            
        2. 二分类交叉熵损失函数（Binary Cross-Entropy Loss）：用于二分类问题，衡量模型预测的概率与真实标签的差异。
            L=-(y*log(y_pred)-(1-y)*log(1-y_pred))               其中y是真实标签=0或1，y_pred是模型预测的概率。
            torch中的二分类交叉熵损失函数：torch.nn.BCELoss()      其中BCELoss是Binary Cross-Entropy Loss的缩写。

    3. 回归问题常用损失函数：
        1. 平均绝对误差损失函数MAE（Mean Absolute Error Loss）也称为L1损失函数：用于回归问题，衡量模型预测值与真实值之间的差异的绝对值均值。
            L=1/n∑(|y-y_pred|)
            torch中的MAE损失函数：torch.nn.L1Loss()
            
            特点：
                1.L1 Loss最大的问题是梯度在零点不平滑，导致容易跳过最极小值，所以不常用
                2.L1 Loss具有稀疏性，为了惩罚较大的值，常常将它作为正则项添加到其它LOSS函数中作为约束

        2. 均方误差损失函数MSE（Mean Squared Error Loss）也称L2损失函数：用于回归问题，衡量模型预测值与真实值之间的差异的平方均值。
            L=1/n∑(y-y_pred)**2
            torch中的MSE损失函数：torch.nn.MSELoss()
            
            特点：
                1.L2 Loss当真实值与目标值相差很大时，梯度就会很大，容易导致模型参数更新缓慢，即容易产生梯度爆炸，所以不常用。
                2.L2 Loss也常作为正则项。

        3.smooth L1损失函数：用于回归问题，是L1损失函数和L2损失函数的结合体，当预测值与真实值差异小时，采用L2损失函数，当差异很大时，采用L1损失函数。
            L=1/n∑[0.5(y-y_pred)**2,|y-y_pred|-0.5]     当|y-y_pred|<=1时使用L2 Loss，当|y-y_pred|>1时使用L1 Loss。
            
            torch中的smooth L1损失函数：torch.nn.SmoothL1Loss()
            
            特点：
                1.smooth L1损失函数在[-1,1]区间内，采用L2损失函数，避免L1跳过极小值问题。
                2.smooth L1损失函数在[-1,1]区间外，采用L1损失函数，所以可以避免L2损失函数在差异很大时的梯度爆炸问题。
'''

'''
各种损失函数的实例：

'''


def cross_entropy_demo():
    '''
    多分类交叉熵损失函数（Cross-Entropy Loss）：用于多分类问题，衡量模型预测的概率分布与真实标签的差异。
    '''
    # 定义真实标签（one-hot编码）
    y_true = torch.tensor([0,1,2],dtype=torch.int64)

    # 定义模型预测的概率分布
    y_pred = torch.tensor([[4,5,7],[8, 9, 16],[1, 9, 14]],dtype=torch.float32)

    # 计算交叉熵损失
    loss = torch.nn.CrossEntropyLoss()
    loss_value = loss(y_pred, y_true)
    print("Cross-Entropy Loss:", loss_value.item())

    # 计算模型预测的类别
    predicted_class = torch.argmax(y_pred, dim=1)
    print("Predicted Class:", predicted_class)
    
def binary_cross_entropy_demo():
    '''
    二分类交叉熵损失函数（Binary Cross-Entropy Loss）：用于二分类问题，衡量模型预测的概率与真实标签的差异。
    '''
    # 定义真实标签（0或1）
    y_true = torch.tensor([0, 1, 1, 0], dtype=torch.float32)

    # 定义模型预测的概率（0到1之间的浮点数）
    y_pred = torch.tensor([0.1, 0.8, 0.9, 0.3], dtype=torch.float32)

    # 计算二分类交叉熵损失
    loss = torch.nn.BCELoss()
    loss_value = loss(y_pred, y_true)
    print("Binary Cross-Entropy Loss:", loss_value.item())

def smooth_l1_loss_demo():
    '''
    平滑L1损失函数（Smooth L1 Loss）：用于回归问题，是L1损失函数和L2损失函数的结合体，当预测值与真实值差异小时，采用L2损失函数，当差异很大时，采用L1损失函数。
    '''
    # 定义真实值和预测值
    y_true = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
    y_pred = torch.tensor([1.2, 1.9, 3.1], dtype=torch.float32)

    # 计算平滑L1损失
    loss = torch.nn.SmoothL1Loss()
    loss_value = loss(y_pred, y_true)
    print("Smooth L1 Loss:", loss_value.item())

    l1=torch.nn.L1Loss()
    l1_value=l1(y_pred,y_true)
    print("L1 Loss:",l1_value.item())

    mse=torch.nn.MSELoss()
    mse_value=mse(y_pred,y_true)
    print("MSE Loss:",mse_value.item())

if __name__=='__main__':
    #cross_entropy_demo()
    #binary_cross_entropy_demo()
    smooth_l1_loss_demo()    