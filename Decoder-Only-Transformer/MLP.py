from torch import nn 


class MLP(nn.Module):
    """
    Arguments: 
    d: Size of embedding dimension 
    bias: whether or not to use bias in linear layer 
    dropout: probability of dropout 
    """
    def __init__(self, d, bias=False, dropout=0.2):

        super().__init__()
        self.c_fc = nn.Linear(d, 4 * d, bias=False)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * d, d, bias=False)
        self.dropout = nn.Dropout(dropout)


    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x