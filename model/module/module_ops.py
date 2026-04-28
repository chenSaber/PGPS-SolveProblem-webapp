import torch.nn as nn

# 这个文件里只有两个非常简单的基础类，
# 它们是搭建大型网络时常用的辅助工具：
# GAP (Global Average Pooling, 全局平均池化)：
# 它的作用是把一个立体的特征图（比如 $C \times W \times H$ 的图像矩阵）拍扁，对每一个通道求平均值，最后变成一个浓缩的向量。
# 我们在看 resnet.py 时提到过，这是一种压缩特征、防止过拟合的常用手段。
# Identity (恒等映射)： 这是一个“什么都不做”的层（输入 $x$，输出 $x$）。它通常在代码框架中充当“占位符”。
# 比如，当我们想通过配置文件动态关闭某个网络模块时，就可以把它替换成 Identity，这样既不改变代码结构，又相当于跳过了这一步。
class GAP(nn.Module):
    """
        Global Average pooling
        Widely used in ResNet, Inception, DenseNet, etc.
     """
    def __init__(self):
        super(GAP, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.avgpool(x)
        # x = x.view(x.shape[0], -1)
        return x


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

