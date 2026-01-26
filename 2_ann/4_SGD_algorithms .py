import torch
import matplotlib.pyplot as plt

'''
八、神经网络的优化方法--->梯度下降算法（Gradient Descent）：
    1. 什么是梯度下降算法？
        1. 梯度下降算法是一种优化算法，用于找到最小化损失函数的方法，即最小的预测值与真实值之间的差。
        2. 梯度下降算法通过迭代更新模型参数，使损失函数值逐渐减小，从而找到最优解。
        3. 样式度下隆算法W(new)=W(old)-n*∂L/∂W    ,其中n是学习率（learning rate），∂L/∂W是损失函数对参数W的梯度。

    2. 梯度下降算法的步骤：
        1. 初始化模型参数（w,b）。
        2. 计算损失函数对参数的梯度（∂L/∂w,∂L/∂b）。
        3. 更新参数（w=w-α*∂L/∂w,b=b-α*∂L/∂b），其中α是学习率（learning rate）。
        4. 重复步骤2和3，直到损失函数值收敛或达到最大迭代次数。

    3. 学习率（Learning Rate）：
        1. 学习率是梯度下降算法中一个超参数，用于控制每次参数更新的步长。
        2. 学习率过大，可能跳过最优解；学习率过小，可能导致模型收敛速度过慢,学习成本高。
        3. 一般选择一个较小的学习率，如0.001、0.01、0.1等。

    4. 梯度下降算法中的四个超参数：
        1. 学习率（Learning Rate）：控制每次参数更新的步长，一般选择一个较小的学习率，如0.001、0.01、0.1等。
        2. Epoch：即指训练轮次，使用全部数据样本，对模型进行一次完整的训练。
        3. Batch：即指训练批次个数
        4. Batch Size：即指每次训练批次样本数量，使用训练集中的小部分样本对模型权重进行更新。
        5. Iterations：即指迭代次数，使用一个Batch Size样本对模型权重进行更新的次数.
        6. 举例:
            1. 假设训练集有1000个样本，选择Batch Size=32（依据GPU内存大小），则批次Batch个数=1000/32=31.25，取整为32。
            2. 每个Epoch需要迭代(Iterations)32次，即一个Epoch需要进行32次权重更新。
            3. 10个Epoch需要迭代320次

    5. 梯度下降的方式：
        1.BGD:批量梯度下降算法（Batch Gradient Descent）：在ANN中不常用，因为计算成本高，内存占用大，当样本数量很大时，计算成本和内存占用会非常高。
            1. 每次迭代使用全部训练样本对模型参数进行更新，即每次迭代使用所有样本的梯度对参数进行更新。
            2. 优点：每次迭代使用全部样本，梯度更准确，模型收敛速度快。
            3. 缺点：计算成本高，内存占用大，当样本数量很大时，计算成本和内存占用会非常高。

        2.SGD:随机梯度下降算法（Stochastic Gradient Descent）：在ANN中不常用，因为每次迭代使用一个样本，梯度估计值不稳定，模型收敛速度慢。
            1. 每次迭代使用一个随机样本对模型参数进行更新，即每次迭代使用一个样本的梯度对参数进行更新。
            2. 优点：每次迭代使用一个样本，计算成本低，内存占用低，当样本数量很大时，计算成本和内存占用会非常低。
            3. 缺点：每次迭代使用一个样本，梯度估计值不稳定，模型收敛速度慢。

        3.Mini-Batch GD:小批量梯度下降算法（Mini-Batch Gradient Descent）：在ANN中常用，因为每次迭代使用一个小批量样本，计算成本低，内存占用低，当样本数量很大时，计算成本和内存占用会非常低。
            1. 每次迭代使用一个小批量训练样本对模型参数进行更新，即每次迭代使用一个Batch Size样本的梯度对参数进行更新。
            2. 优点：每次迭代使用一个小批量样本，计算成本低，内存占用低，当样本数量很大时，计算成本和内存占用会非常低。
            3. 缺点：每次迭代使用一个小批量样本，梯度估计值不稳定，模型收敛速度慢。

    6. 梯度下降算法出现的几个问题以及优化的方法：
        1. 局部最优解问题：梯度下降算法可能会停留在局部最优解，而不是全局最优解。
        2. 鞍点问题：梯度为0的点，模型可能会停留在鞍点，而不是全局最优解。
        3. 梯度碰到平缓区：梯度值比较小，参数优化比较缓慢，可能导致模型收敛时间过长。
        4. 学习率问题：学习率过大，可能跳过最优解；学习率过小，可能导致模型收敛速度过慢,学习成本高。
        5. 批量大小问题：批量大小过小，可能导致模型收敛速度过慢；批量大小过大，可能导致模型内存占用高。

        优化方法--->指数加权平均：
            指数加权平均公式 S_t=β*S_{t-1}+(1-β)*Y_t            
                1. 公式解释：
                    1. S_t：第t次迭代的参数值。
                    2. β：衰减因子，一般选择一个较小的值，如0.9、0.95、0.99等。
                    3. S_{t-1}：第t-1次迭代的参数值。
                    4. Y_t：第t次迭代的梯度值。
                2. 优点：
                    1. 能够加速模型收敛，避免局部最优解问题。
                    2. 能够处理稀疏数据，避免梯度为0的问题。
                3. 缺点：
                    1. 初始学习率较大，历史梯度较大时，学习率较小。
                    2. 学习率过小时，可能导致模型收敛速度过慢。

        优化方法具体实现：
            1.Momentum：动量梯度下降算法（Momentum Gradient Descent）：在ANN中常用，因为它能够加速模型收敛，避免局部最优解问题。
                1. 动量梯度下降公式：
                    1. V_t=β*V_{t-1}+(1-β)*Y_t
                    2. S_t=S_{t-1}-η*V_t
                2. 优点：
                    1. 能够加速模型收敛，避免局部最优解问题。
                    2. 能够处理稀疏数据，避免梯度为0的问题。
                3. 缺点：
                    1. 初始学习率较大，历史梯度较大时，学习率较小。
                    2. 学习率过小时，可能导致模型收敛速度过慢。


            2.Adagrad：Adagrad是一种自适应学习率的优化算法，它根据参数的历史梯度信息，动态调整学习率。


            3.RMSprop：RMSprop是一种自适应学习率的优化算法，优化了Adagrad的问题，避免了学习率过小时的问题。
                1. 使用指数加权平均公式计算参数的历史梯度。
                2. 对参数的更新公式：S_t=S_{t-1}-η*V_t/(sqrt(S_{t-1})+ϵ)

            4.Adam：Adam是一种自适应学习率的优化算法，它结合了Adagrad和RMSprop的优点，能够自动调整学习率，避免学习率过小时的问题。
                1. 每个参数的学习率根据其历史梯度的平方和和历史梯度的一阶矩估计进行调整，初始学习率较大，历史梯度较大时，学习率较小。
                2. 优点：适用于处理稀疏数据，能够自动调整学习率，避免学习率过小时的问题。
                3. 缺点：学习率过小时，可能导致模型收敛速度过慢；学习率过大时，可能跳过最优解。

        学习率衰减方式：
            1.等间隔学习率衰减
            2.指定间隔学习率衰减
            3.指数学习率衰减,一般不建议使用,因为学习率会指数级衰减,可能导致模型收敛速度过慢。
'''


'''
1.指数加权平均实例
2.神经网络优化->各种梯度下降算法的->优化实例
3.学习率衰减实例
'''

def exponential_weighted_average_demo():
    '''
    指数加权平均实例：
        1. 公式：S_t=β*S_{t-1}+(1-β)*Y_t
        2. 优点：
            1. 能够加速模型收敛，避免局部最优解问题。
            2. 能够处理稀疏数据，避免梯度为0的问题。
        3. 缺点：
            1. 初始学习率较大，历史梯度较大时，学习率较小。
            2. 学习率过小时，可能导致模型收敛速度过慢。
    '''
    t=torch.randint(1,40,[30])
    print("原始张量:",t)

    days=torch.arange(1,31,1)
    print("天数:",days)
    # plt.plot(days,t)
    # plt.scatter(days,t)
    # plt.xlabel("Days")
    # plt.ylabel("Temperature")
    # plt.title("Exponential Weighted Average")
    # plt.show()

    #构建指数加权平均算法
    t_avg=[]
    β=0.9
    for i,temp in enumerate(t):
        if i==0:
            t_avg.append(temp)
            continue
        else:
            t2=β*t_avg[i-1]+(1-β)*temp
            t_avg.append(t2)
    
    print("指数加权平均张量:",t_avg)

    # 绘制指数加权平均后的曲线，与原始张量对比展示，
    # 但指数加权平均后的曲线，更平滑，能够更好地反映趋势。
    
    plt.plot(days,t_avg)
    plt.scatter(days,t)
    plt.xlabel("Days")
    plt.show()


def momentum_demo():
    '''
    动量梯度下降算法（Momentum Gradient Descent）：在ANN中常用，因为它能够加速模型收敛，避免局部最优解问题。
    '''
    # 初始w权重参数为=1.0
    w=torch.tensor([1.0],requires_grad=True,dtype=torch.float32)

    # 构建动量梯度下降优化器
    optimizer=torch.optim.SGD([w],      # w权重参数
                              lr=0.01,  # 学习率
                              momentum=0.9 # 动量参数，即β=beta
                             )
    
    # 第1轮计算梯度更新参数
    loss=((w**2)*0.5).sum()     #使用自定义损失函数

    # 梯度清零
    optimizer.zero_grad()

    # 反向传播计算梯度
    loss.backward()

    # 更新参数
    optimizer.step()
    print("w的梯度1:",w.grad)
    print("w的更新值1:",w.detach())

    # 第2轮计算梯度更新参数
    loss=((w**2)*0.5).sum()    #使用自定义损失函数

    # 梯度清零
    optimizer.zero_grad()

    # 反向传播计算梯度
    loss.backward()

    # 更新参数
    optimizer.step()
    print("w的梯度2:",w.grad)
    print("w的更新值2:",w.detach())


def schedule_lr():
    '''
    几种学习率衰减方法：
        1.等间隔学习率衰减
        2.指定间隔学习率衰减
        3.指数学习率衰减
    '''    
    # 参数初始化
    lr=0.1      # 初始学习率
    iter=100    # 每个轮次的迭代次数
    epoche=200  # 总轮次

    # 网格数据初始化
    x=torch.tensor([1.0])   # 输入值
    w=torch.tensor([1.0],requires_grad=True,dtype=torch.float32)    # 权重参数
    y=torch.tensor([1.0])   # 目标值

    # 优化器
    optimizer=torch.optim.SGD([w],      # w权重参数
                              lr=lr,  # 学习率
                              momentum=0.9 # 动量参数，即β=beta
                             )

    # 学习率策略:等间隔衰减
    scheduler_lr_01=torch.optim.lr_scheduler.StepLR(optimizer,    # 优化器
                                                  step_size=50, # 学习率衰减间隔
                                                  gamma=0.5     # 学习率衰减因子
                                                 )
    
    # 学习率策略:指定间隔衰减
    scheduler_lr_02=torch.optim.lr_scheduler.MultiStepLR(optimizer, # 优化器
                                                      milestones=[50,100,150], # 学习率衰减间隔
                                                      gamma=0.5     # 学习率衰减因子
                                                     )
    
    # 学习率策略:指数衰减
    scheduler_lr=torch.optim.lr_scheduler.ExponentialLR(optimizer, # 优化器
                                                          gamma=0.95 # 学习率衰减因子
                                                         )
    
    # 遍历轮次
    epoch_list=[]
    lr_list=[]
    for e in range(epoche):
        # 遍历batch
        epoch_list.append(e)
        lr_list.append(scheduler_lr.get_last_lr())
        for i in range(iter):
            #计算损失
            loss=((w*x-y)**2)*0.5     #使用自定义损失函数

            #更新参数
            optimizer.zero_grad()
            # 反向传播计算梯度
            loss.backward()
            # 更新参数
            optimizer.step()

        #更新学习率lr
        scheduler_lr.step()

    # 展示结果
    plt.plot(epoch_list,lr_list)
    plt.xlabel("Epoch")
    plt.ylabel("Learning Rate")
    plt.title("Learning Rate Schedule")
    plt.show()



def adam_demo():
    pass


def adagrad_demo():
    '''
    Adagrad是一种自适应学习率的优化算法，它根据参数的历史梯度信息，动态调整学习率。
    '''
    # 初始w权重参数为=1.0
    w=torch.tensor([1.0],requires_grad=True,dtype=torch.float32)

    # 构建Adam优化器
    optimizer=torch.optim.Adam([w],     # w权重参数
                               lr=0.01 # 学习率                               
                              )
    
    # 第1轮计算梯度更新参数,更新学习率
    loss=((w**2)*0.5).sum()     #使用自定义损失函数

    # 梯度清零
    optimizer.zero_grad()

    # 反向传播计算梯度
    loss.backward()

    # 更新参数
    optimizer.step()
    print("w的梯度1:",w.grad)
    print("w的更新值1:",w.detach())

    # 第2轮计算梯度更新参数
    loss=((w**2)*0.5).sum()     #使用自定义损失函数

    # 梯度清零
    optimizer.zero_grad()

    # 反向传播计算梯度
    loss.backward()

    # 更新参数
    optimizer.step()
    print("w的梯度2:",w.grad)
    print("w的更新值2:",w.detach())



def rmsprop_demo():
    '''    
    RMSprop是一种自适应学习率的优化算法，优化了Adagrad的问题，避免学习率过快衰减。
    '''
    # 初始w权重参数为=1.0
    w=torch.tensor([1.0],requires_grad=True,dtype=torch.float32)

    # 构建RMSprop优化器
    optimizer=torch.optim.RMSprop([w],     # w权重参数
                               lr=0.01, # 学习率                               
                               alpha=0.9 # 衰减率                               
                              )
    
    # 第1轮计算梯度更新参数,更新学习率
    loss=((w**2)*0.5).sum()     #使用自定义损失函数
    # 梯度清零
    optimizer.zero_grad()

    # 反向传播计算梯度
    loss.backward()

    # 更新参数
    optimizer.step()
    print("w的梯度1:",w.grad)
    print("w的更新值1:",w.detach())
    # 第2轮计算梯度更新参数
    loss=((w**2)*0.5).sum()     #使用自定义损失函数
    # 梯度清零
    optimizer.zero_grad()

    # 反向传播计算梯度
    loss.backward()

    # 更新参数
    optimizer.step()
    print("w的梯度2:",w.grad)
    print("w的更新值2:",w.detach())



if __name__=='__main__':
    #exponential_weighted_average_demo()
    #momentum_demo()
    #adam_demo()
    #adagrad_demo()
    #rmsprop_demo()
    schedule_lr()
