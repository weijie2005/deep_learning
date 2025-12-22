import torch
import torch.nn as nn

torch.random.manual_seed(22)


def view_demo():
    # 原形状
    x = torch.randint(1,10,[3,4,2])
    print("原张量:",x)    
    
    x_view = x.view(3,2,4)
    print("view是否连续:",x_view.is_contiguous())

    print("view后的形状:",x_view.shape)
    print("view后的张量:",x_view)    
   
    x_permute = x.permute(2,0,1)
    print("permute后的形状:",x_permute.shape)
    print("permute后的张量:",x_permute)    
    print("permute是否连续:",x_permute.is_contiguous())

    x_contiguous = x_permute.contiguous()
    print("contiguous是否连续:",x_contiguous.is_contiguous())

def reshape_demo():
    # 原形状
    x = torch.randint(1,10,[3,4,2])
    print("原张量:",x)    
    x_sum = x.sum(dim=1)
    print("\n第0维度求各元素和:",x_sum)
    #print("第1维度:",x[1],"\n第1维度求各元素和:",x.sum(1))
    #print("第2维度:",x[2],"\n第2维度求各元素和:",x.sum(dim=2))
    

    #改变形状
    x_reshaped = x.reshape(-1)
    print("reshape后的形状:",x_reshaped.shape)  
    print("reshape后的张量:",x_reshaped)    

    print("原张量的维数:",x.ndim)
    print("reshape后的维数:",x_reshaped.ndim)    

def squeeze_demo():
    # 移除所有大小为1的维度
    x = torch.randn(1, 2, 1, 3)
    print("原形状:",x.shape)
    print("原张量:",x)

    x_sq = x.squeeze()
    print("删除所有维度为1后的形状:",x_sq.shape)  # torch.Size([2, 3])
    print("删除所有维度为1后的张量:",x_sq)    

    #squeeze(dim=None) 用于移除张量中所有 / 指定维度为 1 的轴
    #dim=int：仅移除指定维度索引序号为int的，（但若该维度≠1，无操作，不报错）
    x_sq = x.squeeze(dim=2)
    print("删除索引dim=1后的形状:",x_sq.shape)  # torch.Size([1，2, 3])
    print("删除索引dim=1后的张量:",x_sq)    

def unsqueeze_demo():
    # 移除所有大小为1的维度
    x = torch.randint(1,10,[3,4,2])
    print("原形状:",x.shape)
    print("原张量:",x)

    x_unsq = x.unsqueeze(dim=2)
    print("在索引dim=2前插入维度1后的形状:",x_unsq.shape)  # torch.Size([1，2, 1, 3])
    print("在索引dim=2前插入维度1后的张量:",x_unsq)    


def transpose_demo():
    # 二维张量转置（等价于 x.t()）
    x = torch.randn(2, 3)
    x_trans = x.transpose(0, 1)
    print("原形状:",x.shape,"->转置后的形状:",x_trans.shape)  # torch.Size([3, 2])

    # 三维张量交换维度
    x_3d = torch.randn(2, 3, 4)  # shape=(2,3,4)
    x_3d_trans = x_3d.transpose(1, 2)  # 交换dim1和dim2
    print("原形状:",x_3d.shape,"->交换dim1和dim2后的形状:",x_3d_trans.shape)  # torch.Size([2, 4, 3])

    # 图像通道转换（HWC → CHW）
    img = torch.randn(28, 28, 3)  # (H,W,C)
    img_chw = img.transpose(2, 0).transpose(1, 2)  # 先交换C和H，再交换W和H
    print("原形状:",img.shape,"->通道转换后的形状:",img_chw.shape)  # torch.Size([3, 28, 28])

    # 转置后张量非连续，需 contiguous() 才能 view()
    new_x=x_trans.contiguous()  # False
    print("原张量:",x_trans,"->转置后的张量:",new_x)

    # 转置后的张量 contiguous() 后可 view()
    print("原形状:",x_trans.shape,"->contiguous 后的形状:",x_trans.contiguous().view(6).shape)  # torch.Size([6])

def permute_demo():
    # 三维张量重排（等价于两次 transpose()）
    img = torch.randint(0, 3, [3, 4, 5])  # (H,W,C)
    img_perm = img.permute(2, 0, 1)  # 新维度顺序：C(2)、H(0)、W(1)
    print("原始形状:",img.shape)
    print("原张量:",img)

    print("permute 后的形状:",img_perm.shape)  # torch.Size([3, 28, 28])（一步完成CHW转换）
    print("permute 后的张量:",img_perm)

    # 四维张量重排
    x_4d = torch.randint(0, 5, [2, 3, 4, 5])  # (N,C,H,W)
    x_4d_perm = x_4d.permute(0, 2, 3, 1)  # (N,H,W,C)
    print("原始形状:",x_4d.shape)
    print("原张量:",x_4d)

    print("permute 后的形状:",x_4d_perm.shape)  # torch.Size([2, 4, 5, 3])
    print("permute 后的张量:",x_4d_perm)

def cut_demo():
    # 原形状
    x1 = torch.randint(1,10,[3,4,2])
    print("原张量:",x1)  

    x2 = torch.randint(1,10,[3,4,2])
    print("原张量:",x2)  

    # 按dim=0拼接
    x_cut = torch.cat([x1,x2],dim=0)
    print("cut后的形状:",x_cut.shape)
    print("cut后的张量:",x_cut)    


def backward_demo():
    # 1.数据：特征张量+目标张量
    x = torch.tensor(5)
    y = torch.tensor(0.)

    # 2.参数：权重张量+偏置张量
    w = torch.tensor(1,requires_grad=True,dtype=torch.float32)
    b = torch.tensor(3,requires_grad=True,dtype=torch.float32)

    #3.前向传播：计算预测值
    y_pred = w * x + b
    print("预测值:",y_pred)
    print("目标值:",y)

    #4.损失函数：计算预测值与目标值的差异
    loss = torch.nn.MSELoss()
    loss_value = loss(y_pred, y)
    print("损失值:",loss_value)
    
    #5.微分，反向传播：计算损失对参数的梯度
    loss_value.backward()    

    #6.打印梯度
    print("w的梯度:",w.grad)
    print("b的梯度:",b.grad)



if __name__ == '__main__':
    #view_demo()    
    #reshape_demo()
    #squeeze_demo()
    #unsqueeze_demo()    
    #transpose_demo()
    #permute_demo()
    #cut_demo()
    backward_demo()