from .resnet import *
from .mobilenet_v2 import *
from config import visual_backbone_list

#它的作用是根据配置文件（args.visual_backbone）的指令，动态决定到底使用哪一种卷积神经网络。
def get_visual_backbone(args):
    if args.visual_backbone in visual_backbone_list:
        model = eval(args.visual_backbone)()
        if args.pretrain_vis_path !="":
            model.load_model(pretrain=args.pretrain_vis_path)
            args.logger.info("Visual backbone has been loaded...")
        else:
            args.logger.info("Visual backbone choose to train from scratch")
        return model
    else:
        raise NotImplementedError("Unsupported Backbone: {}".format(args.visual_backbone))