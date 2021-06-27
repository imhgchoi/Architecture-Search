import os
import time
import uuid
import logging
import shutil
import torch
import random
import numpy as np



def set_seeds(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

def set_paths(args):
    dirs = ['out/', 'logs/', 'save/']
    for root in dirs :
        if not os.path.exists(root):
            os.makedirs(root)
    for path in dirs :
        path += args.name
        if not os.path.exists(path) :
            os.makedirs(path)
        if 'save' in path :
            if not os.path.exists(path+'/runs') :
                os.makedirs(path+'/runs')
            if not os.path.exists(path+'/params') :
                os.makedirs(path+'/params')

def set_logger(args):
    name = args.name
    model_id = time.strftime("%Y%m%d-") + str(uuid.uuid4())[:8]
    file_path = args.logs_path+name.lower()+'_'+model_id+'.log'

    formatter = logging.Formatter('%(asctime)s: %(message)s ', '%m/%d/%Y %I:%M:%S %p')
    logger = logging.getLogger(model_id)
    logger.setLevel(logging.INFO)
    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(formatter)
    logger.addHandler(streamHandler)
    if args.save_log:
        fileHandler = logging.FileHandler(file_path)
        fileHandler.setFormatter(formatter)
        logger.addHandler(fileHandler)
        logger.info('log file : ' + file_path)
    logger.info(args)
    logger.propagate=False
    return logger

def elapsed_time(start, end=None):
    if end is None :
        end = time.time()
    days, rem = divmod(end-start, 86400)
    hours, rem = divmod(rem, 3600)
    minutes, seconds = divmod(rem, 60)
    msg = "elapsed time = {:0>2}d {:0>2}:{:0>2}:{:05.2f}".format(
        int(days), int(hours), int(minutes), seconds)
    return msg
        



################### CLASS OBJECTS

class AverageMeter():
    """ Computes and stores the average and current value """
    def __init__(self):
        self.reset()

    def __call__(self):
        return self.avg

    def reset(self):
        """ Reset all statistics """
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """ Update statistics """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


