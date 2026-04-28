#test.py：期末考试卷（测试与验证逻辑）
# 这正是我们之前费了九牛二虎之力去 Debug 的文件，你应该对它非常熟悉了。
# 它的核心作用就是关闭模型的学习能力（model.eval() 和 with torch.no_grad():），让模型在没见过的题目上真正考一次试。
# 它会调用我们之前修好的判卷模块（compute_exp_result_topk 等），算出最终的答案准确率（Ans_Acc）和公式准确率（Eq_Acc）。

import time
from utils import *
import torch


def validate(args, val_loader, model, tgt_lang):
    batch_time = AverageMeter('Time', ':5.3f')
    acc_ans = AverageMeter('Ans_Acc', ':5.4f')
    acc_eq = AverageMeter('Eq_Acc', ':5.4f')
    progress = ProgressMeter(len(val_loader), [batch_time, acc_ans, acc_eq], args, prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (diagrams, text_dict, var_dict, exp_dict) in enumerate(val_loader):
            # set cuda for input data
            diagrams = diagrams.cuda()
            set_cuda(text_dict)
            set_cuda(var_dict)
            set_cuda(exp_dict)

            # 【核心】调用原作者写好的优雅推理接口
            output = model(diagrams, text_dict, var_dict, exp_dict, is_train=False)

            if args.eval_method == "completion":
                acc1, acc2 = compute_exp_result_comp(output, var_dict, exp_dict, tgt_lang)
            elif args.eval_method == "choice":
                acc1, acc2 = compute_exp_result_choice(output, var_dict, exp_dict, tgt_lang)
            elif args.eval_method == "top3":
                acc1, acc2 = compute_exp_result_topk(output, var_dict, exp_dict, tgt_lang, k_num=3)

            if torch.distributed.is_initialized():
                torch.distributed.barrier()
                reduced_acc_ans = reduce_mean(torch.tensor([acc1]).cuda(), args.nprocs)
                reduced_acc_eq = reduce_mean(torch.tensor([acc2]).cuda(), args.nprocs)
            else:
                reduced_acc_ans = torch.tensor([acc1]).cuda()
                reduced_acc_eq = torch.tensor([acc2]).cuda()

            acc_ans.update(reduced_acc_ans.item(), len(diagrams))
            acc_eq.update(reduced_acc_eq.item(), len(diagrams))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # 【新增】每验证 100 个 batch，打印一次当前平均分，让你不用干等！
            if i % 100 == 0:
                progress.display(i)

    return acc_ans.avg, acc_eq.avg