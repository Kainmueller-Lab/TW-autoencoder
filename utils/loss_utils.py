'''
This file is for losses 
'''
import torch
from torch import nn
import sys
import torch.nn.functional as F

def compute_supervised_classification_loss(predict, target,device):
    num_classes=target.shape[-1]
    weight = torch.ones(num_classes).to(device)
    weight[0] = 0
    loss= F.multilabel_soft_margin_loss(predict,target.float(),weight=weight)
    return loss
