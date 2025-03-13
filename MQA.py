# 导入torch库
import torch
# 从torch库中导入神经网络模块nn
from torch import nn


# 定义多查询注意力模块，继承自torch.nn.Module
class MutiQueryAttention(torch.nn.Module):
    # 初始化函数，hidden_size为隐藏层大小，num_heads为注意力头的数量
    def __init__(self, hidden_size, num_heads):
        # 调用父类的初始化方法
        super(MutiQueryAttention, self).__init__()
        # 保存注意力头的数量
        self.num_heads = num_heads
        # 计算每个注意力头的维度（假设hidden_size可以被num_heads整除）
        self.head_dim = hidden_size // num_heads

        ## 初始化Q、K、V的线性投影层
        # 定义用于生成查询向量的全连接层，输入和输出的维度均为hidden_size
        self.q_linear = nn.Linear(hidden_size, hidden_size)
        # 定义用于生成键向量的全连接层，输出维度为head_dim
        self.k_linear = nn.Linear(hidden_size, self.head_dim)  ###
        # 定义用于生成值向量的全连接层，输出维度为head_dim
        self.v_linear = nn.Linear(hidden_size, self.head_dim)  ###

        ## 初始化输出全连接层，用于整合各注意力头的输出
        self.o_linear = nn.Linear(hidden_size, hidden_size)

    # 定义前向传播函数，hidden_state为输入的隐藏状态，attention_mask为可选的注意力掩码
    def forward(self, hidden_state, attention_mask=None):
        # 获取批次大小，从hidden_state的第一个维度获得
        batch_size = hidden_state.size()[0]

        # 通过q_linear全连接层生成查询向量
        query = self.q_linear(hidden_state)
        # 通过k_linear全连接层生成键向量
        key = self.k_linear(hidden_state)
        # 通过v_linear全连接层生成值向量
        value = self.v_linear(hidden_state)

        # 将查询向量拆分为多个注意力头
        query = self.split_head(query)
        # 将键向量拆分为多个注意力头，传入head_num参数为1
        key = self.split_head(key, 1)
        # 将值向量拆分为多个注意力头，传入head_num参数为1
        value = self.split_head(value, 1)

        ## 计算注意力分数
        # 计算查询和键向量的点积，并对最后一个维度进行转置，再除以head_dim的平方根进行缩放
        attention_scores = torch.matmul(query, key.transpose(-1, -2)) / torch.sqrt(torch.tensor(self.head_dim))

        # 如果提供了注意力掩码，则将其应用于注意力分数（乘以一个很小的负数因子）
        if attention_mask != None:
            attention_scores += attention_mask * -1e-9

        ## 对注意力分数进行归一化
        # 对注意力分数沿着最后一个维度使用softmax函数归一化，得到注意力概率
        attention_probs = torch.softmax(attention_scores, dim=-1)

        # 用归一化的注意力概率对值向量进行加权求和，得到注意力输出
        output = torch.matmul(attention_probs, value)

        # 将输出张量的最后两个维度进行转置，调用contiguous保证内存连续性，
        # 再reshape为(batch_size, 序列长度, head_dim * num_heads)
        output = output.transpose(-1, -2).contiguous().view(batch_size, -1, self.head_dim * self.num_heads)

        # 将整合后的输出通过输出全连接层进行最后的线性变换
        output = self.o_linear(output)

        # 返回最终的注意力输出
        return output

    # 定义辅助函数split_head，用于将输入张量拆分成多个注意力头
    def split_head(self, x, head_num=None):
        # 获取批次大小，从x的第一个维度获得
        batch_size = x.size()[0]

        # 如果未指定head_num，则使用初始化时定义的num_heads进行拆分
        if head_num == None:
            # 将x重塑为 (batch_size, 序列长度, num_heads, head_dim) 并交换第1和第2个维度
            return x.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        else:
            # 如果指定了head_num，则将x重塑为 (batch_size, 序列长度, head_num, head_dim) 并交换第1和第2个维度
            return x.view(batch_size, -1, head_num, self.head_dim).transpose(1, 2)


if __name__ == '__main__':
    # 定义输入张量hidden_state，大小为(2, 4, 8)
    hidden_state = torch.randn(2, 4, 8)
    # 定义注意力掩码张量attention_mask，大小为(2, 4, 4)
    attention_mask = torch.randn(2, 4, 4)

    # 实例化MutiQueryAttention模块，隐藏层大小为8，注意力头数量为2
    attention = MutiQueryAttention(8, 2)
    # 调用MutiQueryAttention的forward方法，传入hidden_state和attention_mask
    output = attention.forward(hidden_state, attention_mask)

    # 打印输出张量的大小
    print(output.size())