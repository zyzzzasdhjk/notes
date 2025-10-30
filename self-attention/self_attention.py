import torch
from torch import nn


class SelfAttentionV1(nn.Module):
    """
    自注意力机制的基础实现版本V1

    自注意力机制允许序列中的每个元素关注序列中其他元素的信息，
    从而捕捉序列内部的依赖关系。该版本实现了最基础的自注意力计算逻辑。
    """
    def __init__(self, embed_size: int):
        """
        初始化自注意力模块

        参数:
            embed_size: 嵌入维度，即每个输入元素的特征维度
        """
        # 调用父类nn.Module的初始化方法
        super().__init__()
        # 保存嵌入维度
        self.embed_size = embed_size

        # 定义查询(Query)、键(Key)、值(Value)的线性投影层
        # 这些投影层将输入特征转换为查询、键、值矩阵
        self.q_proj = nn.Linear(embed_size, embed_size)  # 查询投影层
        self.k_proj = nn.Linear(embed_size, embed_size)  # 键投影层
        self.v_proj = nn.Linear(embed_size, embed_size)  # 值投影层

    def forward(self, X: torch.Tensor):
        """
        前向传播计算自注意力

        参数:
            X: 输入张量，形状为 [batch_size, seq_len, embed_size]
               其中batch_size为批次大小，seq_len为序列长度，embed_size为嵌入维度

        返回:
            注意力计算结果，形状与输入X相同 [batch_size, seq_len, embed_size]
        """
        # 通过线性投影得到查询、键、值矩阵
        # 形状均为 [batch_size, seq_len, embed_size]
        q = self.q_proj(X)  # 查询矩阵 (Query)
        k = self.k_proj(X)  # 键矩阵 (Key)
        v = self.v_proj(X)  # 值矩阵 (Value)

        # 计算注意力分数：查询与键的点积
        # k.transpose(-1, -2) 将键矩阵的最后两个维度交换，形状变为 [batch_size, embed_size, seq_len]
        # 点积结果形状为 [batch_size, seq_len, seq_len]，其中(i,j)位置表示第i个元素对第j个元素的注意力分数
        attention_value = q @ k.transpose(-1, -2)

        # 对注意力分数进行缩放，除以嵌入维度的平方根
        # 目的是防止嵌入维度较大时，点积结果过大导致softmax梯度消失
        attention_value = attention_value / (self.embed_size ** 0.5)

        # 对注意力分数应用softmax激活函数，在最后一个维度上进行归一化
        # 得到注意力权重，形状仍为 [batch_size, seq_len, seq_len]，权重和为1
        weights = torch.softmax(attention_value, dim=-1)

        # 注意力权重与值矩阵相乘，得到最终的注意力输出
        # 形状为 [batch_size, seq_len, embed_size]
        attention_value = weights @ v

        return attention_value


class SelfAttentionV2(nn.Module):
    def __init__(self, embed_size: int):
        super().__init__()
        self.embed_size = embed_size

        # 改进点1：将V1中独立的q_proj、k_proj、v_proj三个线性层合并为一个线性层qkv
        # 作用：一次性计算查询、键、值的拼接结果（输出维度为embed_size*3），减少了线性层的数量
        # 优势：相比三个独立线性层，参数数量相同（均为embed_size*(embed_size)*3 + 3*embed_size），但计算时只需一次矩阵乘法，更高效
        self.qkv = nn.Linear(embed_size, embed_size * 3)

    def forward(self, X: torch.Tensor):
        # 改进点2：通过单个线性层一次性得到q、k、v的拼接张量
        qkv = self.qkv(X)  # 形状：[batch_size, seq_len, embed_size*3]

        # 改进点3：使用split_with_sizes将拼接张量分割为q、k、v三个独立张量
        # 替代了V1中分别通过三个线性层计算q、k、v的方式，功能等价但更简洁
        q, k, v = torch.split_with_sizes(
            qkv, [self.embed_size, self.embed_size, self.embed_size], dim=-1
        )  # 分割后q、k、v形状均为：[batch_size, seq_len, embed_size]

        # 以下注意力计算逻辑与V1完全一致，无改进
        attention_value = q @ k.transpose(-1, -2)
        attention_value = attention_value / (self.embed_size**0.5)
        weights = torch.softmax(attention_value, dim=-1)
        attention_value = weights @ v
        return attention_value

class SelfAttentionV3(nn.Module):
    def __init__(self, embed_size: int, dropout_rate: float = 0.1):
        super().__init__()
        self.embed_size = embed_size
        self.qkv = nn.Linear(embed_size, embed_size * 3)
        self.dropout = nn.Dropout(dropout_rate)
        self.output = nn.Linear(embed_size, embed_size)

    def forward(self, X: torch.Tensor, mask: torch.Tensor | None):
        qkv: torch.Tensor = self.qkv(X)
        q, k, v = torch.split_with_sizes(
            qkv, [self.embed_size, self.embed_size, self.embed_size], dim=-1
        )

        attention_value = q @ k.transpose(-1, -2)
        attention_value: torch.Tensor = attention_value / (self.embed_size**0.5)
        if mask is not None:
            attention_value = attention_value.masked_fill(
                mask == 0, float("-inf")
            )
        attention_value = torch.softmax(attention_value, dim=-1)
        attention_value = self.dropout(attention_value)

        attention_value = attention_value @ v

        return self.output(attention_value)

class SelfAttentionFinal(nn.Module):
    def __init__(self, hiden_dim: int , dropout_rate:float = 0.1):
        super().__init__()
        self.hiden_dim = hiden_dim

        self.query_proj = nn.Linear(hiden_dim, hiden_dim)
        self.key_proj = nn.Linear(hiden_dim, hiden_dim)
        self.value_proj = nn.Linear(hiden_dim, hiden_dim)

        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, X: torch.Tensor , mask : torch.Tensor | None):
        # 先通过矩阵运算得到qkv矩阵
        q : torch.Tensor = self.query_proj(X)
        k : torch.Tensor = self.key_proj(X)
        v : torch.Tensor = self.value_proj(X)

        # 然后计算 q @ k
        attention_value = q @ k.transpose(-1,-2)
        attention_value = attention_value / (self.hiden_dim ** 0.5)
        
        # 进行掩码
        if mask:
            attention_value = attention_value.masked_fill(
                mask == 0 , 1e-20
            )
        
        # 继续计算公式
        
        attention_value = self.dropout(attention_value)
        attention_value = torch.softmax(attention_value , dim = -1)
        attention_value = attention_value @ v

        return attention_value
