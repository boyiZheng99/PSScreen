import math
import numpy as np

import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score, roc_auc_score, \
    confusion_matrix
    
class AverageMeter(object):
    """Compute current value, sum and average"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val, self.avg, self.sum, self.count = 0, 0, 0, 0

    def update(self, val, num=1):
        self.val = val
        self.sum += val
        self.count += num
        self.avg = float(self.sum) / self.count if self.count != 0 else 0
        
        
