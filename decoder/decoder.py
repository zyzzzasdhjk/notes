import torch.nn as nn


class SimpleDecoderLayer(nn.Module):
    def __init__(self, hidden_dim , head_num , attentionn_dropout_rate=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.head_dim = hidden_dim // head_num

        # mha
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.o_proj = nn.Linear(hidden_dim, hidden_dim)
        self.drop_att = nn.Dropout(attentionn_dropout_rate)
        self.att_ln = nn.LayerNorm(hidden_dim,eps=1e-6)

        # ffn (升维->降维->LayerNorm)
        self.up_proj = nn.Linear(hidden_dim, hidden_dim * 4)
        self.down_proj = nn.Linear(hidden_dim * 4, hidden_dim)
        self.ffn_ln = nn.LayerNorm(hidden_dim,eps=1e-6)
        self.act_fn = nn.GELU()
        self.drop_ffn = nn.Dropout(attentionn_dropout_rate)
    
    def mha(self,X,mask=None):
        batch , seq = X.size()
        
        return X
    
    def ffn(self,X):
        up = self.up_proj(X)
        up = self.act_fn(up)
        down = self.down_proj(up)
        down = self.drop_ffn(down)
        X = self.ffn_ln(X + down)
        return X

    def forward(self, X, attn_mask = None):
        # mha
        X = self.mha(X,attn_mask)
        X = self.ffn(X)