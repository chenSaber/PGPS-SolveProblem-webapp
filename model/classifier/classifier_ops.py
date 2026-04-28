#classifier_ops.py：四大决策引擎（核心重点！）
# 这个文件里定义了 4 种具体的分类（打分）算法。为什么一个简单的输出预测需要写 4 种方法？
# 这其实涉及到了深度学习中非常硬核的**“长尾分布（类别不平衡）”问题。
# 在几何解题（PGPS9K 数据集）中，像 Sum（求和）、Equal（相等）这种算子可能出现几万次，
# 而 RNgon_H_Area（正多边形面积）可能只出现几十次。
# 如果用普通的分类器，模型就会变成一个“势利眼”，疯狂偏袒常见的算子，遇到罕见定理就全当没看见。
# 为了解决这个问题，原作者准备了以下四个“武器”：

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# 第一种：DotProduct (经典点积分类器)算法思路：
# 也就是最基础的全连接层（nn.Linear）。计算公式是经典的 $y = Wx + b$。作用：
# 这是深度学习的默认配置。它直接将模型最后输出的高维特征向量，通过矩阵乘法（点积）映射到词汇表的维度。
# 分数最高的那一维，就是模型预测的单词。

class DotProduct(nn.Module):
    def __init__(self, num_classes=1000, feat_dim=2048, bias=True):
        super(DotProduct, self).__init__()
        # print('<DotProductClassifier> contains bias: {}'.format(bias))
        self.fc = nn.Linear(feat_dim, num_classes,bias)
        
    def forward(self, x, *args):
        x = self.fc(x)
        return x

# 第三种：CosNorm (余弦相似度分类器)算法思路： 它是 FCNorm 的一种变体，源自人脸识别领域（如 CosFace）。
# 它利用数学中的余弦公式，计算特征向量 $x$ 和各个类别权重 $w$ 之间的夹角余弦值。
# 作用： 相比于算距离，算“角度”能让同一类的特征在空间上聚得更紧，不同类的特征离得更远，从而提高模型在分类字典算子时的“判别力”，
# 减少张冠李戴（比如把 Sin_Law 预测成 Cos_Law）的概率。

class CosNorm(nn.Module):
    def __init__(self, in_dims, out_dims, scale=16, margin=0.5, init_std=0.001):
        super(CosNorm, self).__init__()
        self.in_dims = in_dims
        self.out_dims = out_dims
        self.scale = scale
        self.margin = margin
        self.weight = nn.Parameter(torch.Tensor(out_dims, in_dims).cuda())
        self.reset_parameters() 

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, input, *args):
        norm_x = torch.norm(input.clone(), 2, 1, keepdim=True)
        ex = (norm_x / (1 + norm_x)) * (input / norm_x)
        ew = self.weight / torch.norm(self.weight, 2, 1, keepdim=True)
        return torch.mm(self.scale * ex, ew.t())

# 第二种：FCNorm (特征与权重双归一化分类器 —— 应对长尾分布的利器！)
# 算法思路： 在代码中有一行非常关键的注释：# for LDAM Loss。
# LDAM（Label-Distribution-Aware Margin）是深度学习界用来对付**“极度不平衡数据”**的著名算法。
# 实现细节： 普通点积中，出现次数多的算子（比如 Equal），其对应的权重向量模长（magnitude）会变得巨大，导致模型总是倾向于输出它。
# FCNorm 通过 F.normalize(x) 和 F.normalize(self.weight)，强制把特征向量和权重向量的长度都压缩为 1。
# 这样大家就不拼“长短”了，纯拼“角度（相似度）”，最后再乘以一个缩放因子 scale。它能极大地保护那些“罕见几何定理”不被模型遗忘。

class FCNorm(nn.Module):
    # for LDAM Loss
    def __init__(self, num_features, num_classes, scale=20.0):
        super(FCNorm, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, num_features))
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)
        self.scale = scale

    def forward(self, x):
        out = self.scale * F.linear(F.normalize(x), F.normalize(self.weight))
        return out

# 第四种：DistFC (欧式距离分类器)算法思路：
# 这里没有使用常规的乘法，而是用了一个初中数学完全平方公式：$(x-c)^2 = x^2 + c^2 - 2xc$。
# 代码中的 dist = features_square + centers_square - features_into_centers 正是这一步的完美复刻。
# 作用： 它为词典里的每一个词设定了一个“中心点（Centers/Prototypes）”。
# 模型的决策方式变成了：我的特征在这个多维空间里离哪个词的中心点最近，我就输出哪个词。
# 这种做法常用于 Center Loss，能强迫模型学到更纯粹的特征表征。

class DistFC(nn.Module):

    def __init__(self, num_features, num_classes,init_weight=True):
        super(DistFC, self).__init__()
        self.centers=nn.Parameter(torch.randn(num_features,num_classes).cuda(),requires_grad=True)
        if init_weight:
            self.__init_weight()

    def __init_weight(self):
        nn.init.kaiming_normal_(self.centers)

    def forward(self, x):
        features_square=torch.sum(torch.pow(x,2),1, keepdim=True)
        centers_square=torch.sum(torch.pow(self.centers,2),0, keepdim=True)
        features_into_centers=2.0*torch.matmul(x, (self.centers))
        dist=features_square+centers_square-features_into_centers   
        return self.centers, dist


