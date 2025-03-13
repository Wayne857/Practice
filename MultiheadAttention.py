import torch
from torch import nn


class MultiheadAttention(nn.Module):
    # n_heads: 多头注意力的数量
    # hid_dim: 每个词输出的向量维度
    def __init__(self, hid_dim, n_heads, dropout):
        super(MultiheadAttention, self).__init__()
        self.hid_dim = hid_dim
        self.n_heads = n_heads

        # 强制hid_dim必须整除n_heads
        assert hid_dim % n_heads == 0
        # 定义W_q, W_k, W_v矩阵
        self.w_q = nn.Linear(hid_dim, hid_dim)
        self.w_k = nn.Linear(hid_dim, hid_dim)
        self.w_v = nn.Linear(hid_dim, hid_dim)
        self.fc = nn.Linear(hid_dim, hid_dim)
        self.do = nn.Dropout(dropout)
        # 缩放
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim//n_heads]))

    def forward(self, query, key, value, mask=None):
        # K: [batch_size, key_len, hid_dim]
        # V: [batch_size, value_len, hid_dim]
        # Q: [batch_size, query_len, hid_dim]
        batch_size = query.shape[0]
        Q = self.w_q(query)
        K = self.w_k(key)
        V = self.w_v(value)

        # 这里把K Q V矩阵拆分成多个头，变成一个四维张量
        # 最后一维是hid_dim//n_heads得到的，表示每组注意力的维度
        # 转置是为了把注意力头放到第一维，方便计算
        Q = Q.view(batch_size, -1, self.n_heads, self.hid_dim //
                   self.n_heads).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.n_heads, self.hid_dim //
                   self.n_heads).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.n_heads, self.hid_dim //
                   self.n_heads).permute(0, 2, 1, 3)

        # 第一步是计算Q和K的点积，除以scale
        attention = torch.matmul(Q, K.permute(0, 1, 3, 2))/self.scale
        # 如果mask不为空，就把mask加进去
        if mask is not None:
            attention = attention.masked_fill(mask == 0, -1e10)

        # 第二步，计算上一步的结果的softmax，再经过dropout，得到attention
        attention = self.do(torch.softmax(attention, dim=-1))

        # 第三步，attention和V相乘，得到多头注意力结果
        x = torch.matmul(attention, V)
        x = x.permute(0, 2, 1, 3).contiguous()

        x = x.view(batch_size, -1, self.hid_dim)
        x = self.fc(x)
        return x


query = torch.rand(64, 12, 300)
key = torch.rand(64, 10, 300)
value = torch.rand(64, 10, 300)
attention = MultiheadAttention(hid_dim=300, n_heads=6, dropout=0.1)
output = attention(query, key, value)

print(output.shape)


