import torch
import torch.nn as nn
import math

# resnet.py：主力视觉网络 (The Workhorse)
# 这是你整个项目最核心的视觉特征提取器。
# ResNet（残差网络）是深度学习图像处理领域的经典之作。
# 核心思想（残差连接）： 在 SimpleBlock 和 BottleneckBlock 的 forward 函数中，你会看到一句非常关键的代码：out = out + short_out。
# 这就是大名鼎鼎的“残差连接（Skip Connection）”。
# 它允许原始图像信息直接跨越某些层传递到后面，完美解决了神经网络太深导致“梯度消失”（也就是模型学不到东西）的问题。
# 你使用的模型：ResNet10： 在代码底部，定义了从 ResNet10 到 ResNet101 的不同版本。
# 根据你在日志中显示的信息，你的项目默认使用的是最轻量级的 ResNet10。它包含 4 个阶段的卷积层（[1, 1, 1, 1]），逐步将输入的彩色几何原图提取为高维的特征图。
# 打平操作 (Flatten)： 经过层层卷积，原本 2D 的图像最终经过 AvgPool2d（平均池化）和 Flatten() 操作，
# 被“压扁”成了一维的特征向量（维度大小存放在 self.final_feat_dim 中，对应 ResNet10 通常是 512 维）。
# 这个一维向量，就是模型对这幅几何图的“视觉理解”。

# 输入阶段： 一张复杂的几何题目图片（张量名为 diagram_src）被送入网络。
# 提取阶段（在这里发生）： 图片进入 self.visual_extractor（也就是这里的 ResNet10），经过几层卷积和残差传递，被压缩成一个 512 维的纯数字向量。
# 特征对齐： 这个视觉向量会经过 self.visual_emb_unify（几个线性层），把维度变换到和文本特征相同的维度（cfg.encoder_embedding_size）。
# 多模态融合： 最后，这个视觉向量会和题目的“文本特征（text_emb_src）”拼接在一起（torch.cat），送入后续的编码器和解码器进行逻辑推理。

def init_layer(L):
    # Initialization using fan-in
    if isinstance(L, nn.Conv2d):
        n = L.kernel_size[0]*L.kernel_size[1]*L.out_channels
        L.weight.data.normal_(0,math.sqrt(2.0/float(n)))
    elif isinstance(L, nn.BatchNorm2d):
        L.weight.data.fill_(1)
        L.bias.data.fill_(0)

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()
        
    def forward(self, x):        
        return x.view(x.size(0), -1)

# Simple ResNet Block
class SimpleBlock(nn.Module):
    maml = False #Default
    def __init__(self, indim, outdim, half_res):
        super(SimpleBlock, self).__init__()
        self.indim = indim
        self.outdim = outdim
        self.C1 = nn.Conv2d(indim, outdim, kernel_size=3, stride=2 if half_res else 1, padding=1, bias=False)
        self.BN1 = nn.BatchNorm2d(outdim)
        self.C2 = nn.Conv2d(outdim, outdim,kernel_size=3, padding=1,bias=False)
        self.BN2 = nn.BatchNorm2d(outdim)
        self.relu1 = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)

        self.parametrized_layers = [self.C1, self.C2, self.BN1, self.BN2]

        self.half_res = half_res

        # if the input number of channels is not equal to the output, then need a 1x1 convolution
        if indim!=outdim:
            self.shortcut = nn.Conv2d(indim, outdim, 1, 2 if half_res else 1, bias=False)
            self.BNshortcut = nn.BatchNorm2d(outdim)
            self.parametrized_layers.append(self.shortcut)
            self.parametrized_layers.append(self.BNshortcut)
            self.shortcut_type = '1x1'
        else:
            self.shortcut_type = 'identity'

        for layer in self.parametrized_layers:
            init_layer(layer)

    def forward(self, x):
        out = self.C1(x)
        out = self.BN1(out)
        out = self.relu1(out)
        out = self.C2(out)
        out = self.BN2(out)
        short_out = x if self.shortcut_type == 'identity' else self.BNshortcut(self.shortcut(x))
        out = out + short_out
        out = self.relu2(out)
        return out

# Bottleneck block
class BottleneckBlock(nn.Module):
    maml = False #Default
    def __init__(self, indim, outdim, half_res):
        super(BottleneckBlock, self).__init__()
        bottleneckdim = int(outdim/4)
        self.indim = indim
        self.outdim = outdim

        self.C1 = nn.Conv2d(indim, bottleneckdim, kernel_size=1,  bias=False)
        self.BN1 = nn.BatchNorm2d(bottleneckdim)
        self.C2 = nn.Conv2d(bottleneckdim, bottleneckdim, kernel_size=3, stride=2 if half_res else 1,padding=1)
        self.BN2 = nn.BatchNorm2d(bottleneckdim)
        self.C3 = nn.Conv2d(bottleneckdim, outdim, kernel_size=1, bias=False)
        self.BN3 = nn.BatchNorm2d(outdim)

        self.relu = nn.ReLU()
        self.parametrized_layers = [self.C1, self.BN1, self.C2, self.BN2, self.C3, self.BN3]
        self.half_res = half_res

        # if the input number of channels is not equal to the output, then need a 1x1 convolution
        if indim!=outdim:
            self.shortcut = nn.Conv2d(indim, outdim, 1, stride=2 if half_res else 1, bias=False)
            self.parametrized_layers.append(self.shortcut)
            self.shortcut_type = '1x1'
        else:
            self.shortcut_type = 'identity'

        for layer in self.parametrized_layers:
            init_layer(layer)

    def forward(self, x):

        short_out = x if self.shortcut_type == 'identity' else self.shortcut(x)
        out = self.C1(x)
        out = self.BN1(out)
        out = self.relu(out)
        out = self.C2(out)
        out = self.BN2(out)
        out = self.relu(out)
        out = self.C3(out)
        out = self.BN3(out)
        out = out + short_out

        out = self.relu(out)
        return out

class ResNet(nn.Module):
    maml = False #Default
    def __init__(self,block,list_of_num_layers, list_of_out_dims, flatten = True):
        # list_of_num_layers specifies number of layers in each stage
        # list_of_out_dims specifies number of output channel for each stage
        super(ResNet,self).__init__()
        assert len(list_of_num_layers)==4, 'Can have only four stages'

        conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                                            bias=False)
        bn1 = nn.BatchNorm2d(64)
        relu = nn.ReLU()
        pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        init_layer(conv1)
        init_layer(bn1)
        trunk = [conv1, bn1, relu, pool1]
        indim = 64
        for i in range(4):
            for j in range(list_of_num_layers[i]):
                half_res = (i>=1) and (j==0)
                B = block(indim, list_of_out_dims[i], half_res)
                trunk.append(B)
                indim = list_of_out_dims[i]
        if flatten:
            avgpool = nn.AvgPool2d(4)
            trunk.append(avgpool)
            trunk.append(Flatten())
            self.final_feat_dim = indim
        else:
            self.final_feat_dim = [indim, 4, 4]
        self.trunk = nn.Sequential(*trunk)

    def forward(self,x):
        out = self.trunk(x)
        return out

def ResNet10(flatten = True):
    return ResNet(SimpleBlock, [1,1,1,1],[64,128,256,512], flatten)

def ResNet18(flatten = True):
    return ResNet(SimpleBlock, [2,2,2,2],[64,128,256,512], flatten)

def ResNet34(flatten = True):
    return ResNet(SimpleBlock, [3,4,6,3],[64,128,256,512], flatten)

def ResNet50(flatten = True):
    return ResNet(BottleneckBlock, [3,4,6,3], [256,512,1024,2048], flatten)

def ResNet101(flatten = True):
    return ResNet(BottleneckBlock, [3,4,23,3],[256,512,1024,2048], flatten)
