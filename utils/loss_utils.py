'''
This file is for losses 
'''
import torch
from torch import nn
import sys
import torch.nn.functional as F
from .boundary_detect import DetectSPBoundary, LocalDiscrepancy



# --------------------------------------
# segementation loss in the train.py
# --------------------------------------
def min_segloss(pred_class_scores, class_gt, heatmaps,seg_gt,seg_gt_exists, device):
    weights = (1-torch.concat((torch.ones((class_gt.size(0),1)).to(self.device),class_gt),dim=1))
    loss=nn.CrossEntropyLoss(weight=weights,reduction="sum")(input=hm_sf*heatmaps,target=0*seg_gt)

    return loss

def BCE_segloss(pred_class_scores, class_gt, heatmaps,seg_gt,seg_gt_exists, device):
    pass

def softmax_segloss(pred_class_scores, class_gt, heatmaps,seg_gt,seg_gt_exists, device):
    pass



def only_bg_segloss(pred_class_scores, class_gt, heatmaps,seg_gt,seg_gt_exists, device):
    L_bg=bg_fg_prob_loss(pred_class_scores, class_gt, heatmaps,seg_gt)
    L_duc=depress_unexist_class_loss(pred_class_scores, class_gt, heatmaps,seg_gt)
    heatmap_loss=L_bg+L_duc

    pass


# --------------------------------------
# Useful functions for designing the loss
# --------------------------------------


class LocalConsistentLoss(nn.Module):
    def __init__(self, in_channels, l_type='l1'):
        super(LocalConsistentLoss, self).__init__()
        self.semantic_boundary = DetectSPBoundary(padding_mode='zeros')
        self.neighbor_dif = LocalDiscrepancy(in_channels=in_channels, padding_mode='replicate', l_type=l_type)

    def forward(self, x, label):
        discrepancy = self.neighbor_dif(x)
        mask = self.semantic_boundary(label)
        
        loss = discrepancy[mask].mean()
        return loss   

def local_consistent_loss(pred_class_scores, class_gt, heatmaps,seg_gt):
    '''
    heatmaps: after softmax operation, value between 0 to 1, shape N x 7 x H x W
    class_gt: shape N x 6 
    '''
    fg_mask=(seg_gt!=0)
    heatmaps=torch.softmax(heatmap,dim=1)

    # experiment with the flag 
    only_exist_class=True
     
    if only_exist_class:
        loss=0
        if class_gt.shape[-1]==7:
            pass
        elif class_gt.shape[-1]==6:
            class_gt=torch.cat(torch.ones((class_gt.shape[0],1)),class_gt)
        else:
            raise NotImplementedError

        for i in heatmaps.shape[0]:
            tmp_heatmaps=(heatmaps[i][class_gt[i]]).unsqueeze(0)
            loss+=LocalConsistentLoss(tmp_heatmaps,fg_mask)
    else:
        loss=LocalConsistentLoss(in_channels=heatmaps.shape[1])(heatmaps,fg_mask)

    return loss


def depress_unexist_class_loss(pred_class_scores, class_gt, heatmaps,seg_gt):
    '''
    heatmaps: after softmax operation, value between 0 to 1, shape N x 7 x H x W
    class_gt: shape N x 6 
    '''
    if class_gt.shape[-1]==7:
        class_gt=class_gt[:,1:]
    elif class_gt.shape[-1]==6:
        pass
    else:
        raise NotImplementedError

    nuclei_prob=nn.Softmax(dim=1)(heatmaps[:,1:,:,:])
    depress_class=1-class_gt
    depress_nuclei_heatmaps=nuclei_prob[depress_class==1]
    loss=torch.mean(depress_nuclei_heatmaps)
    return loss


def bg_fg_prob_loss(pred_class_scores, class_gt, heatmaps,seg_gt):
    '''
    heatmaps: value between -inf to +inf, shape N x 7 x H x W
    '''
    bg_mask=(seg_gt==0).float() # N x H x W  the GT mask for bg
    fg=torch.sum(heatmaps[:,1:,:,:],dim=1)
    hetmap_scale=1000
    bg_fg_prob=nn.Softmax(dim=1)(torch.stack((heatmap_scale*heatmaps[:,0,:,:], fg),dim=1))

    # input shape N x C x H x W, target N x H x W
    loss=nn.CrossEntropyLoss(reduction="sum")(input=bg_fg_prob,target=(1-bg_mask).long())
    return loss
    
    
