# __init__.py：教鞭选择器
# 这个文件极其简单，就是一个工厂函数 get_criterion(args)。它根据你的配置文件 args.criterion，决定在当前训练中抽出哪一种“教鞭”去惩罚模型。
from .loss import *
from config import criterion_list


def get_criterion(args):   
    # create model
    if args.criterion in criterion_list:
        return eval(args.criterion)(args)
    else:
        raise NotImplementedError("Unsupported Loss Criterion : {}".format(args.criterion)) 