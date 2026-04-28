# attention.py：解码器的“聚光灯”与“准星”（核心重点！）
# 这里面定义了三种注意力机制（Attention）。在 Seq2Seq（序列到序列）的几何解题模型中，模型需要一边生成解题指令，一边回头看题目。这就需要用到注意力机制。
# 这三个类本质上都是 Bahdanau 注意力（加性注意力） 的变体：
# 第一种：Attn —— 决定“眼睛看向哪里”的聚光灯
# 作用： 当解码器（GRU）准备生成下一步操作时（比如决定这一步要用勾股定理），它会处于一个特定的“内部思考状态”（代码里的 hidden）。
# Attn 类会把当前的 hidden 状态和编码器处理好的所有图文特征（encoder_outputs）进行对比打分。
# 结果：经过 Softmax 函数后，它会输出一个概率分布（attn_energies）。概率越高的部分，说明模型当前越应该“聚焦”在那段文本或那块图像区域上。
# 第二种：Score —— 决定“从哪里拿变量”的指针网络（Pointer Network）
# 作用： 这个极其关键！它就是任务书中提到的“自限制解码”的具体实现。当模型决定了要用 Gougu 算子后，接下来要填入参数（比如填入哪两条边）。
# 模型不能瞎编数字，只能从候选列表（候选池）里挑。
# 逻辑： 1.  输入 hidden（当前想找变量的需求）和 candi_embeddings（这道题里出现的所有已知数字和中间变量的特征）。
# 2.  torch.cat 拼接它们，通过神经网络计算出一个匹配得分。
# 3.  核心操作 masked_fill_(~candi_mask, -1e12)： 如果某个词不在当前题目的候选列表里，代码会直接把它的得分变成负无穷大（-1e12）。
# 这就是“限制输出空间”！这样模型最终挑选时，绝对不可能选到题目里没出现过的幽灵数字。
# 第三种：Score_Multi —— 并行的批量打分器
# 作用： 它是 Score 的升级版。Score 是“想好了一步，去挑一次候选词”。
# 而 Score_Multi 支持输入一个序列的 hidden，同时对多个步骤、多个候选词进行批量的矩阵打分。
# 这主要是为了在训练阶段利用 GPU 的并行计算能力，大幅度提高运算速度。

import torch
import torch.nn as nn

class Score(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Score, self).__init__()
        self.attn = nn.Linear(hidden_size + input_size, hidden_size)
        self.score = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, hidden, candi_embeddings, candi_mask=None):
        '''
        Arguments:
            hidden: B x 1 x 2H
            candi_embeddings: B x candi_size x H
            candi_mask: B x candi_size
        Return:
            score: B x candi_size
        '''
        hidden = hidden.repeat(1, candi_embeddings.size(1), 1)  # B x candi_size x H
        # For each position of encoder outputs
        energy_in = torch.cat((hidden, candi_embeddings), 2)  # B x candi_size x 3H
        score = self.score(torch.tanh(self.attn(energy_in))).squeeze(-1)  # B x candi_size
        if candi_mask is not None:
            score = score.masked_fill_(~candi_mask, -1e12)
        return score

class Attn(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Attn, self).__init__()
        self.attn = nn.Linear(hidden_size + input_size, hidden_size)
        self.score = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, hidden, encoder_outputs, seq_mask=None):
        '''
        Arguments:
            hidden: B x 1 x H (q)
            encoder_outputs: B x S x H
            seq_mask: B x S
        Return:
            attn_energies: B x S
        '''
        hidden = hidden.repeat(1, encoder_outputs.size(1), 1)  # B x S x H
        energy_in = torch.cat((hidden, encoder_outputs), 2) # B x S x 2H
        score_feature = torch.tanh(self.attn(energy_in)) # B x S x H
        attn_energies = self.score(score_feature).squeeze(-1)  # B x S
        if seq_mask is not None:
            attn_energies = attn_energies.masked_fill_(~seq_mask, -1e12)
        attn_energies = nn.functional.softmax(attn_energies, dim=1)  # B x S

        return attn_energies
    
class Score_Multi(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Score_Multi, self).__init__()
        self.attn = nn.Linear(hidden_size + input_size, hidden_size)
        self.score = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, hidden, candi_embeddings, candi_mask=None):
        '''
        Arguments:
            hidden: B x S x H
            candi_embeddings: B x candi_size x H
            candi_mask: B x candi_size
        Return:
            score: B x S x candi_size
        '''
        hidden = hidden.unsqueeze(2).repeat(1, 1, candi_embeddings.size(1), 1) # B x S x candi_size x H
        candi_embeddings = candi_embeddings.unsqueeze(1).repeat(1, hidden.size(1), 1, 1) # B x S x candi_size x H
        candi_mask = candi_mask.unsqueeze(1).repeat(1, hidden.size(1), 1) # B x S x candi_size
        energy_in = torch.cat((hidden, candi_embeddings), -1)  # B x S x candi_size x 2H
        score = self.score(torch.tanh(self.attn(energy_in))).squeeze(-1)  # B x S x candi_size
        if candi_mask is not None:
            score = score.masked_fill_(~candi_mask, -1e12)
        return score