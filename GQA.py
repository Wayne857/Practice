## 分组注意力查询
import torch
from torch import nn

# 定义一个GroupQueryAttention类，继承自nn.Module
class GroupQueryAttention(torch.nn.Module):
    def __init__(self, hidden_size, num_heads, group_num):
        super(GroupQueryAttention, self).__init__()

        # 设置头数、每个头的维度和组数
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.group_num = group_num

        # 初始化Q、K、V投影矩阵
        self.q_linear = nn.Linear(hidden_size, hidden_size)  # 查询矩阵Q
        self.k_linear = nn.Linear(hidden_size, self.group_num * self.head_dim)  # 键矩阵K
        self.v_linear = nn.Linear(hidden_size, self.group_num * self.head_dim)  # 值矩阵V

        # 输出的线性变换层
        self.o_linear = nn.Linear(hidden_size, hidden_size)

    # 定义前向传播函数
    def forward(self, hidden_state, attention_mask=None):
        batch_size = hidden_state.size()[0]  # 获取批次大小

        # 计算Q、K、V
        query = self.q_linear(hidden_state)  # 计算查询向量Q
        key = self.k_linear(hidden_state)  # 计算键向量K
        value = self.v_linear(hidden_state)  # 计算值向量V

        # 将Q、K、V拆分成多个头
        query = self.split_head(query)
        key = self.split_head(key, self.group_num)  # 按照组数拆分键
        value = self.split_head(value, self.group_num)  # 按照组数拆分值

        # 计算注意力分数
        attention_scores = torch.matmul(query, key.transpose(-1, -2)) / torch.sqrt(torch.tensor(self.head_dim))

        # 如果提供了attention_mask，则对注意力分数做遮盖
        if attention_mask != None:
            attention_scores += attention_mask * -1e-9

        # 对注意力分数进行softmax归一化，得到注意力权重
        attention_probs = torch.softmax(attention_scores, dim=-1)

        # 根据注意力权重加权求值
        output = torch.matmul(attention_probs, value)

        # 对输出进行维度转换，并恢复原始形状
        output = output.transpose(-1, -2).contiguous().view(batch_size, -1, self.head_dim * self.num_heads)

        # 通过输出线性层映射到最终的输出空间
        output = self.o_linear(output)

        return output

    # 定义拆分头部的函数
    def split_head(self, x, group_num=None):

        # 获取批次大小和序列长度
        batch_size, seq_len = x.size()[:2]

        # 如果没有给定group_num，按照头数拆分
        if group_num == None:
            return x.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        else:
            # 按照给定的组数拆分
            x = x.view(batch_size, -1, group_num, self.head_dim).transpose(1, 2)
            # 扩展x的维度并重新排列，以符合多头注意力的需求
            x = x[:, :, None, :, :].expand(batch_size, group_num, self.num_heads // group_num, seq_len, self.head_dim).reshape(batch_size, self.num_heads // group_num * group_num, seq_len, self.head_dim)
            return x

if __name__ == '__main__':
    # 定义输入张量
    input_tensor = torch.randn(32, 20, 512)

    # 定义多头注意力层
    multihead_attn = GroupQueryAttention(512, 8, 4)

    # 输出多头注意力层的结果
    output_tensor = multihead_attn(input_tensor)
    print(output_tensor.size())  # torch.Size([32, 20, 512])