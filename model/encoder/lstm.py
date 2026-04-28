#lstm.py：备用编码器
# 这是 GRU 的“老大哥” LSTM（长短期记忆网络）。
# 作用： 它的存在通常是为了做“消融实验（对比实验）”。
# 在你的毕业论文中，如果你想证明“我的模型用 GRU 效果好”，
# 你就可以把配置文件改成 lstm 跑一遍，然后对比准确率。一般来说，GRU 参数更少、收敛更快，而 LSTM 计算更复杂。

import torch.nn as nn


class LSTM(nn.Module):
    
    def __init__(self, cfg):
        super(LSTM, self).__init__()

        self.lstm = nn.LSTM(
            input_size=cfg.WORD_EMBED_SIZE,
            hidden_size=cfg.HIDDEN_SIZE, # int(hidden_size / num_directions),
            num_layers=cfg.NUM_LAYERS,
            batch_first=cfg.BATCH_FIRST,  # first dim is batch_size or not
            bidirectional=cfg.BIDIRECTIONAL
        )

    def forward(self, input, h0, c0):
        output, (hn, cn) = self.lstm(input, (h0, c0))
        return output, hn, cn




