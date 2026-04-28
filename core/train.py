#train.py：魔鬼训练营（训练逻辑）
#这个文件只有短短几十行，但它包含了深度学习模型学习知识的核心循环。
#每次它会从 train_loader 里拿出一个 Batch 的题目（图文、变量字典、答案字典）。
#让模型去蒙答案（output = model(...)）。
#用损失函数（criterion）计算模型蒙的答案和真实答案之间的差距。
#最核心的三连击： optimizer.zero_grad(), loss.backward(), optimizer.step()。通过反向传播算法，模型会根据刚才犯的错，自动调整自己内部的几千万个参数，让自己下次变得更聪明。
import time
from utils import *

def train(args, epoch, train_loader, model, criterion, optimizer):

    batch_time = AverageMeter('Time', ':5.3f')
    data_time = AverageMeter('Data', ':5.3f')
    losses = AverageMeter('Loss', ':.4e')
    progress = ProgressMeter(len(train_loader), [batch_time, data_time, losses],
                             args, prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()
    end = time.time()

    for i, (diagrams, text_dict, var_dict, exp_dict) in enumerate(train_loader):
        '''
            text_dict = {'token', 'sect_tag', 'class_tag', 'len'}
            var_dict = {'pos', 'len', 'var_value', 'arg_value'}
            exp_dict = {'exp', 'len', 'answer'}
        '''
        # measure data loading time
        data_time.update(time.time() - end)
        # set cuda for input data
        diagrams = diagrams.cuda()
        set_cuda(text_dict), set_cuda(var_dict), set_cuda(exp_dict)
        # compute output
        output = model(diagrams, text_dict, var_dict, exp_dict, is_train=True)
        loss = criterion(output, exp_dict['exp'][:,1:].clone(), exp_dict['len']-1) # Remove special symbol [SOS]
        # update the loss
        torch.distributed.barrier()
        reduced_loss = reduce_mean(loss, args.nprocs)
        losses.update(reduced_loss.item(), len(diagrams))
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if i % args.print_freq == 0:
            progress.display(i, lr = optimizer.state_dict()['param_groups'][0]['lr'])

    return losses.avg
