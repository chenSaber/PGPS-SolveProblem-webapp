# gru.py：双向融合引擎（当前项目的主力核心！）
# 这是你当前模型真正在使用的多模态融合大脑。
# 算法思路： 这是一个**双向（Bidirectional）**的 GRU（门控循环单元）。
# 普通单向网络只能从左读到右，而双向网络就像是一个人把题目“正着读一遍，反着再读一遍”，这样它就能同时获取上下文的完整语义。
# 神仙级代码细节 (pack_padded_sequence)： 你会在 forward 函数里看到这个 PyTorch 的高级操作。
# 因为几何题目有长有短，为了统一形状，短的题目后面会被补上很多 0（Padding）。如果不加处理，模型会傻乎乎地去计算这些 0，既浪费时间又干扰结果。
# pack_padded_sequence 就是告诉 GPU：“这些零不要算，直接跳过！”这极大地提升了训练效率。
# 融合操作： 仔细看这一句 pade_outputs[:, :, :self.hidden_size] + pade_outputs[:, :, self.hidden_size:]。
# 由于是双向的，网络会输出“正向”和“反向”两个特征。代码将它们直接相加（特征融合），浓缩成了模型对当前图文信息的最终理解。

import torch.nn as nn


class GRU(nn.Module):

    def __init__(self, cfg):
        super(GRU, self).__init__()

        self.is_bidirectional = True
        self.batch_first = True
        self.gru = nn.GRU(
            input_size = cfg.encoder_embedding_size,
            hidden_size = cfg.encoder_hidden_size, # int(hidden_size / num_directions),
            num_layers = cfg.encoder_layers,
            bidirectional = self.is_bidirectional,
            dropout = cfg.dropout_rate,
            batch_first = self.batch_first
        )
        self.hidden_size = cfg.encoder_hidden_size
        self.dropout = nn.Dropout(cfg.dropout_rate)
    
    def forward(self, src_emb, input_lengths, hidden=None):

        input_emb = self.dropout(src_emb)
        # input_emb = src_emb
        packed = nn.utils.rnn.pack_padded_sequence(input_emb, input_lengths.cpu(), \
                                            batch_first=self.batch_first, enforce_sorted=False)
        pade_hidden = hidden
        pade_outputs, pade_hidden = self.gru(packed, pade_hidden)
        pade_outputs, _ = nn.utils.rnn.pad_packed_sequence(pade_outputs, batch_first=self.batch_first)
        # pade_outputs [B, S, hidden_size*num_directions] 
        # pade_hidden [n_layers*num_directions, B, hidden_size]
        if self.is_bidirectional: 
            pade_outputs = pade_outputs[:, :, :self.hidden_size] + pade_outputs[:, :, self.hidden_size:]  # B x S x H
            pade_hidden = pade_hidden[0::2, :, :] + pade_hidden[1::2, :, :]

        return pade_outputs, pade_hidden




