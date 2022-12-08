import torch
import os.path as osp
import time
import sys
import logging
import torch.distributed as dist
import os

#  由于使用了 yolov5 所以要对handle 修改并且不绑定屏幕handle 
# def setup_logger(logpth):
#     logfile = 'Myattack-{}.log'.format(time.strftime('%Y-%m-%d-%H-%M-%S'))
#     logfile = osp.join(logpth, logfile)
#     print(logfile)
#     FORMAT = '%(levelname)s %(filename)s(%(lineno)d): %(message)s'
#     log_level = logging.INFO
#     if dist.is_initialized() and not dist.get_rank()==0:
#         log_level = logging.ERROR
#     logging.basicConfig(level=log_level, format=FORMAT, filename=logfile)
#     logging.root.addHandler(logging.StreamHandler())

class Log:
    def __init__(self, file_name):
        # 第一步，创建一个logger
        self.logger = logging.getLogger(file_name)  # file_name为多个logger的区分唯一性
        self.logger.setLevel(logging.DEBUG)  # Log等级总开关
        # 如果已经有handler，则用追加模式，否则直接覆盖
        mode = 'a' if self.logger.handlers else 'w'
        # 第二步，创建handler，用于写入日志文件和屏幕输出
        log_path = '/data1/yjt/adversarial_attack/myattack/logs/'
        # log_path = os.getcwd() + '/'
        logfile = log_path + file_name + '.txt'
        #
        fmt = "%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s"
        formatter = logging.Formatter(fmt)
        # 文件输出
        fh = logging.FileHandler(logfile, mode=mode)
        fh.setLevel(logging.DEBUG)  # 输出到file的log等级的开关
        fh.setFormatter(formatter)
        # 往屏幕上输出
        # sh = logging.StreamHandler()
        # sh.setFormatter(formatter)  # 设置屏幕上显示的格式
        # sh.setLevel(logging.DEBUG)
        # 先清空handler, 再添加
        self.logger.handlers = []
        self.logger.addHandler(fh)
        # self.logger.addHandler(sh)

    def info(self, message):
        self.logger.info(message)


if __name__=='__main__':
    # logger = logging.getLogger()
    # setup_logger('/data1/yjt/adversarial_attack/myattack/logs')
    # logger.info('this is my try')
    # logger.info('this is my try2')
    log = Log('b')
    log.info('我是bbbbb')
    log1 = Log('c')
    log1.info('我是ccccc')
    log.info('我还是bbb')
    log2 = Log('d')
    log2.info('我是dddd')