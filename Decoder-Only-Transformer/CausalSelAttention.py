import math
import torch
from torch import nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):

    def __init__(self, d, H, T, bias=False, dropout=0.2):
        """
        Arguments:
        d: Size of embedding dimensions
        H: Number of attention heads
        T: Maximum length of input sequence (in tokens)
        bias: whether or not to use bias in linear layers
        dropout: probability of dropout
        """
        super().__init__()
        assert d % H == 0

        # key, query, value projection for all heads but in a batch
        self.c_attn = nn.Linear(d, 3 * d, bias=bias)
        self.c_proj = nn.Linear(d, d, bias=bias)  # ← missing earlier!

        # Dropout modules
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        self.H = H
        self.d = d

        # Causal mask to ensure attention only looks left
        self.register_buffer("mask", torch.tril(torch.ones(T, T)).view(1, 1, T, T))

    def forward(self, x):
        B, T, _ = x.size()  # (batch, seq_len, embed_dim)

        # Project and split into q, k, v
        q, k, v = self.c_attn(x).split(self.d, dim=2)

        k = k.view(B, T, self.H, self.d // self.H).transpose(1, 2)
        q = q.view(B, T, self.H, self.d // self.H).transpose(1, 2)
        v = v.view(B, T, self.H, self.d // self.H).transpose(1, 2)

        # Attention computation
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))  # [B, H, T, T]
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        # Apply attention to values
        y = att @ v  # [B, H, T, d//H]
        y = y.transpose(1, 2).contiguous().view(B, T, self.d)

        # Final projection
        y = self.resid_dropout(self.c_proj(y))
        return y
    

if __name__ == "__main__":
    torch.manual_seed(42)

    # Parameters 
    batch_size = 2 
    seq_len = 5 
    embed_dim = 8 
    num_heads = 2 
    max_seq_len = 10 

    # Create model 
    attn = CausalSelfAttention(d=embed_dim, H=num_heads, T=max_seq_len, dropout=0.1)

    # Dummy input: (batch_size, seq_len, embed_dim)
    x = torch.randn(batch_size, seq_len, embed_dim)

    # Forward pass 
    out = attn(x)

    print("Input shape: ", x.shape)
    print("Output shape: ", out.shape)
    print("Output tensor: ", out)