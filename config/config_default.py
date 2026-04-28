#config_default.py：项目的“总开关台”与“参数面板”
# 这是整个项目的配置核心，里面使用了 Python 的 argparse 库定义了所有的超参数（Hyperparameters）。你在写论文“实验环境与参数设置”一节时，所有的数据都要从这里抄。
# 架构选择开关： * visual_backbone：决定用 ResNet10 还是 mobilenet_v2。
# encoder_type：决定用 gru、lstm 还是 transformer。
# decoder_type：决定用 rnn_decoder 还是 tree_decoder。
# 我们在之前的 __init__.py 文件里看到的那些“动态调度”，全都是根据这里设定的字符串来执行的。
# 尺寸与维度调节： 例如 diagram_size (128)、encoder_embedding_size (256)、encoder_hidden_size (512) 等。这些参数决定了模型“大脑”的容量有多大。
# 训练策略配置： 包含了我们排错时见过的 use_MLM_pretrain、预训练权重路径 MLM_pretrain_path，以及优化器选项、评估方法 eval_method 等等。

import argparse
import torchvision.models as models


model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__") and callable(models.__dict__[name]))
                   
criterion_list = ["CrossEntropy", "FocalLoss", "MaskedCrossEntropy"]
optimizer_list = ["SGD", "ADAM"]
scheduler_list = ["multistep",'cosine','warmup']
visual_backbone_list = ['ResNet10', 'mobilenet_v2']
encoder_list = ['lstm', 'gru', 'transformer']
decoder_list = ["rnn_decoder", "tree_decoder"]
eval_method_list = ["completion", "choice", "top3"]
dataset_list = ['Geometry3K', 'PGPS9K'] 

def get_parser():
    parser = argparse.ArgumentParser(description='PyTorch PGPS Training')

    #网络架构旋钮 (Architecture)
    # --visual_backbone (如 'ResNet10'): 决定用什么模型来当“眼睛”。ResNet10 比较轻量，跑得快；如果换成 ResNet50 眼睛会更亮，但吃显存。
    # --encoder_type (如 'gru'): 决定用什么模型当“大脑”处理文本与图像的融合序列。这里用到的是我们之前提到的 Bi-GRU。
    # --decoder_type: 决定用什么方式输出公式序列（也就是“嘴巴”）。
    # --encoder_embedding_size / --encoder_hidden_size: 这些数字（如 256, 512）决定了模型内部张量（Tensor）的维度。数字越大，模型脑容量越大，但越容易过拟合且训练越慢。

    # visual backbone       视觉模型相关
    ##############################################################################
    parser.add_argument('--visual_backbone', default="ResNet10", type=str, choices=visual_backbone_list)
    parser.add_argument('--diagram_size',  default=128, type=int)
    # encoder model         编码器（文本/序列编码）
    ##############################################################################
    parser.add_argument('--encoder_type', default="gru", type=str, choices=encoder_list)
    parser.add_argument('--encoder_layers', default=2, type=int)
    parser.add_argument('--encoder_embedding_size', default=256, type=int)
    parser.add_argument('--encoder_hidden_size', default=512, type=int)
    parser.add_argument('--max_input_len', default=400, type=int)
    # decoder model     解码器
    ##############################################################################
    parser.add_argument('--decoder_type', default="rnn_decoder", type=str, choices=decoder_list)
    parser.add_argument('--decoder_layers', default=2, type=int)
    parser.add_argument('--decoder_embedding_size', default=512, type=int)
    parser.add_argument('--decoder_hidden_size', default=512, type=int)
    parser.add_argument('--max_output_len', default=40, type=int)
    # general model     通用模型设置
    ##############################################################################
    parser.add_argument('--dropout_rate', default=0.2, type=float)
    parser.add_argument('--beam_size', default=10, type=int)
    # optimizer     优化器与训练调度
    ##############################################################################
    parser.add_argument('--optimizer_type', default="ADAMW", type=str, choices=optimizer_list)
    parser.add_argument('--lr', default=1e-3, type=float, help='initial learning rate without LM')
    parser.add_argument('--lr_LM', default=1e-4, type=float, help='initial learning rate of LM')
    parser.add_argument('--weight_decay', default=0.01, type=float)
    parser.add_argument('--max_epoch', default=540, type=int)
    parser.add_argument('--scheduler_type', default="warmup", type=str, choices=scheduler_list)
    parser.add_argument('--scheduler_step', default=[160, 280, 360, 440, 500], type=list)
    parser.add_argument('--scheduler_factor', default=0.5, type=float, help='learning rate decay factor')
    parser.add_argument('--cosine_decay_end', default=0.0, type=float, help='cosine decay end')
    parser.add_argument('--warm_epoch', default=40, type=int)

    #训练策略旋钮 (Training Strategy)
    # --criterion (如 'MaskedCrossEntropy'): 损失函数，也就是教鞭。决定了模型做错题时怎么惩罚它（我们之前聊过，掩码交叉熵专门为了屏蔽 Padding）。
    # --optimizer (如 'ADAM'): 优化器。决定了模型发现错误后，以什么姿势去调整自己的参数。
    # --batch_size (如 32): 批次大小。决定了模型每次看几道题再更新一次参数。设置得越大，训练越稳定，但需要极大的显卡内存（显存不够会报 OOM 错误）。
    # --max_epoch: 决定了模型要把整个训练集（8000多道题）反复刷多少遍。

    # criterion     损失函数与评估方式
    ###############################################################################
    parser.add_argument('--criterion', default="MaskedCrossEntropy", choices=criterion_list, type=str)
    parser.add_argument('--eval_method', default="top3", choices=eval_method_list, type=str)
    # dataset       数据集与预训练
    ################################################################################
    parser.add_argument('--dataset', default="PGPS9K", type=str, choices=dataset_list)
    parser.add_argument('--dataset_dir', default='./datasets/PGPS9K_all')
    parser.add_argument('--pretrain_vis_path', default='')
    parser.add_argument('--vocab_src_path', default='./vocab/vocab_src.txt')
    parser.add_argument('--vocab_tgt_path', default='./vocab/vocab_tgt.txt')
    parser.add_argument('--pretrain_emb_path', default='')
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--random_prob', default=0.5, type=float)
    parser.add_argument('--without_stru', action='store_true', help='structure clauses are used or not')
    parser.add_argument('--trim_min_count', default=5, type=int, help='minimum number of word')
    parser.add_argument('--use_MLM_pretrain', action='store_true', help='use MLM pretrain')
    parser.add_argument('--MLM_pretrain_path', default='./pretraining_model/LM_MODEL.pth')
    # print information日志与评估频率
    ###################################################################################
    parser.add_argument('--dump_path', default="./log/", type=str, help='save log path')
    parser.add_argument('--print_freq', default=20, type=int, help='print frequency')
    parser.add_argument('--eval_epoch', default=40, type=int)
    # general config训练通用参数
    ###################################################################################
    parser.add_argument('--workers', default=4, type=int)
    parser.add_argument('--evaluate_only', action='store_true', help='evaluate model on validation set')
    parser.add_argument('--resume_model', default="", type=str, help='use pre-trained model')
    # DistributedDataParallel分布式训练相关
    ###################################################################################
    parser.add_argument('--local_rank', default=0, type=int, help='node rank for distributed training')
    parser.add_argument('--init_method', default="env://", type=str, help='distributed init method')
    parser.add_argument('--debug', action='store_true', help = "if debug than set local rank = 0")
    parser.add_argument('--seed', default=202302, type=int,help='seed for initializing training. ')

    return parser.parse_args()
