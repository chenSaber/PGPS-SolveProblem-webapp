#lr_scheduler.py：学习率调度器（模型训练的“变速箱”）
# 这里定义了一个自定义的调度器 WarmupMultiStepLR。
# Warmup（热身阶段）： 在训练刚开始的前几个 Epoch（如 warmup_epochs=5），模型还是一片空白，
# 如果一上来就用很大的学习率（步子迈太大），很容易“扯着蛋”导致梯度爆炸。所以它使用线性或常数的方法，让学习率从很小慢慢爬升到正常值。
# MultiStep（阶梯降速）： 等过了热身期，在达到特定的里程碑（milestones）时，它会把学习率乘以一个衰减因子（gamma，通常是 0.1 或 0.5）。
# 这就像开车快到目的地时需要减速一样，能帮助模型在训练后期更精细地寻找最优解。

import torch
from bisect import bisect_right


class WarmupMultiStepLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
            self,
            optimizer,
            milestones,
            gamma=0.1,
            warmup_factor=1.0 / 3,
            warmup_epochs=5,
            warmup_method="linear",
            last_epoch=-1,
    ):
        if not list(milestones) == sorted(milestones):
            raise ValueError(
                "Milestones should be a list of" " increasing integers. Got {}",
                milestones,
            )

        if warmup_method not in ("constant", "linear"):
            raise ValueError(
                "Only 'constant' or 'linear' warmup_method accepted"
                "got {}".format(warmup_method)
            )
        self.milestones = milestones
        self.gamma = gamma
        self.warmup_factor = warmup_factor
        self.warmup_epochs = warmup_epochs
        self.warmup_method = warmup_method
        super(WarmupMultiStepLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        warmup_factor = 1
        if self.last_epoch < self.warmup_epochs:
            if self.warmup_method == "constant":
                warmup_factor = self.warmup_factor
            elif self.warmup_method == "linear":
                alpha = float(self.last_epoch) / self.warmup_epochs
                warmup_factor = self.warmup_factor * (1 - alpha) + alpha
        return [
            base_lr
            * warmup_factor
            * self.gamma ** bisect_right(self.milestones, self.last_epoch)
            for base_lr in self.base_lrs
        ]
