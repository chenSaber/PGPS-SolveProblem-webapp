# loss.py：三大惩罚机制（核心）
# 原作者在这里准备了三种不同严厉程度的损失函数：
# 第一种：CrossEntropy (普通交叉熵)
# 这是深度学习分类任务中最经典的损失函数。
# 作用： 直接衡量模型输出的概率分布与真实标签之间的差距。但它有一个缺点：在面对长短不一的句子时，它会把用来补齐长度的无用填充符（Padding [PAD]）也算进惩罚里，这显然是不公平的。
# 第二种：FocalLoss (焦点损失)
# 这是一个专门用来对付**“严重偏科”（长尾分布）**的“定制教鞭”。
# 算法思路： 在几何解题里，像“相加（Sum）”这种算子太常见了，模型一学就会；而“正多边形面积”这种算子可能半天才出现一次，模型总是学不会。
# FocalLoss 通过一个动态缩放因子 (1 - p) ** self.gamma，让模型**“少管那些已经稳拿高分的简单题，把所有的学习精力（Loss 惩罚）全部集中在那些总是做错的罕见难题上”**。
# 第三种：MaskedCrossEntropy (掩码交叉熵 —— 当前主力！)
# 这是自然语言处理（NLP）和序列生成任务中绝对的标配，也是你的模型默认在用的损失函数。
# 核心机制 mask = sequence_mask(length)： 因为每道几何题需要的解题步骤长短不一，短的步骤后面会被强制塞满 [PAD]。如果让模型因为猜错了 [PAD] 而挨打，模型就会学傻。
# 作用： 这个函数巧妙地引入了一个“掩码（Mask）”。在计算总分时：losses = losses * mask.float()。
# 这行代码就像是涂卡笔，直接把所有 [PAD] 对应的惩罚值涂成了 0。这样，模型只需要为它真正生成的有意义的算子负责。
import torch
import torch.nn as nn
from torch.nn import functional as F
from utils import *


class CrossEntropy(nn.Module):
    def __init__(self, cfg):
        super(CrossEntropy, self).__init__()

    def forward(self, output, target):
        loss = F.cross_entropy(output, target)
        return loss

class FocalLoss(nn.Module):
    def __init__(self, cfg=None):
        super(FocalLoss, self).__init__()
        # self.gamma = cfg.LOSS.FOCAL.GAMMA
        if cfg is None:
            self.gamma = 2.0
        else:
            self.gamma = cfg.focal_loss_gamma
        assert self.gamma >= 0

    def focal_loss(self, input_values):
        """Computes the focal loss"""
        p = torch.exp(-input_values)
        loss = (1 - p) ** self.gamma * input_values
        return loss.mean()

    def forward(self, input, target):
        return self.focal_loss(F.cross_entropy(input, target, reduction='none'))

class MaskedCrossEntropy(nn.Module):

    def __init__(self, cfg):
        super(MaskedCrossEntropy, self).__init__()
        self.cfg = cfg
    
    def forward(self, logits, target, length):
        """
        Args:
            logits: A Variable containing a FloatTensor of size
                (batch, max_len, num_classes) which contains the
                unnormalized probability for each class.  B x S x (op_size+const_size+var_size)
            target: A Variable containing a LongTensor of size
                (batch, max_len) which contains the index of the true
                class for each corresponding step. B x S
        Returns:
            loss: An average loss value masked by the length.
        """
        # logits_flat: (batch * max_len, num_classes)
        logits_flat = logits.view(-1, logits.size(-1)) 
        # log_probs_flat: (batch * max_len, num_classes)
        log_probs_flat = F.log_softmax(logits_flat, dim=1)
        # target_flat: (batch * max_len, 1)
        target_flat = target.view(-1, 1)
        # losses_flat: (batch * max_len, 1)
        losses_flat = -torch.gather(log_probs_flat, dim=1, index=target_flat)
        # losses: (batch, max_len)
        losses = losses_flat.view(*target.size())
        # mask: (batch, max_len)
        mask = sequence_mask(length)
        losses = losses * mask.float()
        loss = losses.sum() / length.float().sum()
        return loss