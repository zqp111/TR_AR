from sklearn.metrics import confusion_matrix
import torch.nn as nn
import torch
import numpy as np
import argparse
from processor.base_method import import_class
import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1, 2, 3'
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


if __name__ == '__main__':
    Processor = import_class('processor.action_recognition.Recognition')
    p = Processor(['--config', './config/train_sub_xp_model.yaml'])   #sys.argv[0]指.py程序本身,argv[2:]指从命令行获取的第二个参数

    print('start')
    p.start()