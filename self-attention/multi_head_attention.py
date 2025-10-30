import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_dim: int, head_num: int, dropout_rate: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.head_num = head_num
        self.head_dim = hidden_dim // head_num # 768 // 8 = 96

        # hidden_dim -> head_dim * head_num
        self.q_proj = nn.Linear(hidden_dim, hidden_dim) 
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, hidden_dim)

        self.attention_dropout = nn.Dropout(dropout_rate)

    def forward(self, X: torch.Tensor, mask: torch.Tensor | None):
        # X shape: (batch_size, seq_len, hidden_dim)
        bathc_size, seq_len, _ = X.shape

        q : torch.Tensor = self.q_proj(X)
        k : torch.Tensor = self.k_proj(X)
        v : torch.Tensor = self.v_proj(X)

        # (b,s,h) -> (b , head_num, s, head_dim)
        q = q.view(bathc_size, seq_len, self.head_num, self.head_dim).transpose(1, 2)
        k = k.view(bathc_size, seq_len, self.head_num, self.head_dim).transpose(1, 2)
        v = v.view(bathc_size, seq_len, self.head_num, self.head_dim).transpose(1, 2)

        attention_weights = torch.matmul(q, k.transpose(-2, -1)) / self.head_dim ** 0.5
        # k.transpose(-2, -1) (b, head_num, head_dim, s)

        if mask is not None: # mask shape: (batch_size, seq_len, seq_len)
            attention_weights = attention_weights.masked_fill(mask == 0, float('-inf'))
        attention_weights = torch.softmax(attention_weights, dim=-1)
        attention_weights = self.attention_dropout(torch.softmax(attention_weights, dim=-1))

        output = torch.matmul(attention_weights, v)

        output = output.transpose(1, 2).contiguous().view(bathc_size, seq_len, self.hidden_dim)

        output = self.output(output)

        return output