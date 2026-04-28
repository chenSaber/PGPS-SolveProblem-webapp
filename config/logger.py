# logger.py：记录一切的“黑匣子”
# 我们在终端里看到的那些带着时间戳的输出，以及生成的 log/2026-02-28... 文件夹，全靠这个文件来运作。
# 双路记录机制： 这个文件封装了 Python 标准的 logging 模块。
# 它同时设置了 StreamHandler（把信息打印到你的屏幕上）和 TimedRotatingFileHandler（把信息偷偷写进硬盘的日志文件里备份）。
# 多卡防刷屏机制： 里面有一句非常关键的代码：if rank!=0: return。
# 因为项目是基于 DistributedDataParallel 多卡分布式训练的，如果 4 张显卡同时向屏幕打印日志，你的终端瞬间就会乱码。这句话保证了永远只有主卡（Rank 0）有资格说话，保持了我们监控日志时的清爽。
import logging
from logging import handlers

class Logger(object):
    level_relations = {
        'debug':logging.DEBUG,
        'info':logging.INFO,
        'warning':logging.WARNING,
        'error':logging.ERROR,
        'crit':logging.CRITICAL
    }

    def __init__(self, filename, rank, level='info', when='D', backCount=3, fmt='%(asctime)s - %(levelname)s: %(message)s'):
        self.logger = logging.getLogger(filename)
        if rank!=0: return 
        format_str = logging.Formatter(fmt)
        self.logger.setLevel(self.level_relations.get(level))
        sh = logging.StreamHandler()
        sh.setFormatter(format_str)
        th = handlers.TimedRotatingFileHandler(filename=filename,when=when,backupCount=backCount,encoding='utf-8')
        th.setFormatter(format_str)
        self.logger.addHandler(sh)
        self.logger.addHandler(th)

def create_logger(filepath, rank):
    log = Logger(filepath, rank)
    return log.logger
