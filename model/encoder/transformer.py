# transformer.py：
# 深层语义提取器Transformer 是当今大语言模型（如 ChatGPT）的基石，这里用它来做文本特征的深层理解。
# 算法思路 - PositionalEncoding（位置编码）： 传统的 Transformer 是不分先后的“词袋”模型。
# 原作者使用了基于 $\sin$ 和 $\cos$ 函数的正弦波位置编码（代码里的 math.log(10000.0) 公式非常经典）。
# 这就给原本零散的单词打上了“位置时间戳”，让模型知道哪句话在前面，哪句话在后面。算法思路 - LearnedPositionEncoding：
# 这是一个非常聪明的微创新！它专门为了处理几何题目中的“变量（var_pos）”设计。
# 它把题目里出现的未知数、线段名（如 AB, CD）等变量所在的位置，通过一个 nn.Embedding 层变成了可学习的向量。
# 算法思路 - TransformerEncoder： 配合掩码（Mask），让模型在看一句话时，能自动找出和当前单词最相关的信息（自注意力机制），从而得出极其深度的语义理解。
import torch
import torch.nn as nn
from utils.utils import sequence_mask
import math

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
            x: [B, max_len, d_model]
            pe: [1, max_len, d_model]
        """
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return self.dropout(x)

class LearnedPositionEncoding(nn.Module):

    def __init__(self, d_model, max_len = 20):
        super(LearnedPositionEncoding, self).__init__()
        self.embedding = nn.Embedding(max_len, d_model)

    def forward(self, x, var_pos):
        """
            x: [B, max_len, d_model]
            var_pos: [B, var_len]
        """
        loc_mat = torch.zeros(x.size(0), x.size(1), dtype=torch.int64).cuda()
        pos_id = torch.arange(1, var_pos.size(1)+1).repeat(var_pos.size(0), 1).cuda()
        pos_id[var_pos==var_pos.min()] = 0
        loc_mat.scatter_(1, var_pos, pos_id)

        x = x + self.embedding(loc_mat)

        return x

class TransformerEncoder(nn.Module):

    def __init__(self, d_model=256, nhead=8, num_encoder_layers=6, dim_feedforward=1024, dropout=0.2):
        super(TransformerEncoder,self).__init__()

        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
        encoder_norm = nn.LayerNorm(d_model)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)
        self.position = PositionalEncoding(d_model=d_model)
        
        self._reset_parameters()
        self.d_model = d_model
        self.nhead = nhead
    
    def _reset_parameters(self):
        """
            Initiate parameters in the transformer model.
        """
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, len_src, emb_src):
        # mask
        src_key_padding_mask = ~sequence_mask(len_src)
        # position encoding
        emb_src = self.position(emb_src) 
        # encoder   
        memory = self.encoder(emb_src.permute(1,0,2), src_key_padding_mask=src_key_padding_mask)

        return memory.permute(1,0,2)