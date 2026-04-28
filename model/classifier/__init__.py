# __init__.py：决策引擎调度中心
# 和 backbone 里的初始化文件一样，这是一个“调度器（Dispatcher）”。
# 由于在不同的实验或任务下，模型需要采取不同的决策策略，这个文件通过读取配置 args.classifier，为你动态挑选四种不同的分类器之一（FCNorm, CosNorm, DotProduct, DistFC）。

from .classifier_ops import *
from config import classifier_list

       
def get_classifier(args):

    bias_flag = args.classifier_bias
    num_features = args.num_features
    num_classes = args.num_classes

    if not args.classifier in classifier_list:
        raise NotImplementedError("Unsupported Classifier: {}".format(args.classifier))

    if args.classifier == "FCNorm":
        classifier = FCNorm(num_features, num_classes)
    elif args.classifier == "CosNorm":
        classifier = CosNorm(num_features, num_classes)
    elif args.classifier == "DotProduct":
        classifier = DotProduct(num_classes, num_features, bias_flag)
    elif args.classifier == "DistFC":
        classifier = DistFC(num_features, num_classes)
 
    return classifier