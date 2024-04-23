import torch 
import sys
from sklearn.metrics import average_precision_score
import numpy as np

from torch.nn import functional as F
# metrics computation from https://github.com/shirgur/AGFVisualization/
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.initialized = False
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None

    def initialize(self, val, weight):
        self.val = val
        self.avg = val
        self.sum = val * weight
        self.count = weight
        self.initialized = True

    def update(self, val, weight=1):
        if not self.initialized:
            self.initialize(val, weight)
        else:
            self.add(val, weight)

    def add(self, val, weight):
        self.val = val
        self.sum += val * weight
        self.count += weight
        self.avg = self.sum / self.count

    def value(self):
        return self.val

    def average(self):
        return self.avg

# --------------------------------------
# Functions for semantic segmentation (AP and pixel accuracy)
# --------------------------------------

def calculate_ap_scores(predict, target, ignore_index=-1):
    '''
    returns the average percision scores
    for prediction scores (NxCxHxW),  tensor on cpu
    and targets (NxHxW) with classes [0,1,2,3,...]

    return:
        average AP scores for N images
        number of images
    '''
    # rewrite to do it in tensor manner 06.04.2023
    total = []
    for pred, tgt in zip(predict, target.detach().cpu()):
        target_expand = tgt.unsqueeze(0).expand_as(pred) #CxHxW
        target_expand_numpy = target_expand.numpy().reshape(-1) #flatten

        # Tensor process
        x = torch.zeros_like(target_expand)
        t = tgt.unsqueeze(0).clamp(min=0).long()
        target_1hot = x.scatter_(0, t, 1)
        predict_flat = pred.numpy().reshape(-1)
        target_flat = target_1hot.data.numpy().reshape(-1)

        p = predict_flat[target_expand_numpy != ignore_index]
        t = target_flat[target_expand_numpy != ignore_index]

        total.append(np.nan_to_num(average_precision_score(t, p)))
    return torch.tensor(total).mean(), len(total)
        
def calculate_pix_accuracy(predict, target):
    '''
    returns the the pixel-wise accuracy
    for prediction scores (NxCxHxW), tensor on cpu
    and targets (NxHxW) with classes [0,1,2,3,...]
    excluding the the bg class 0
    '''
    # rewrite to do it in tensor manner 06.04.2023
 
    _, predict = torch.max(predict, 1) # N x H x W
    predict = predict.numpy() # N x H x W
    target = target.detach().cpu().numpy() # N x H x W
    pixel_labeled = np.sum((target > 0)) # total numbers of pixels which is not bg class for N images
    pixel_correct = np.sum((predict == target) * (target > 0))
    assert pixel_correct <= pixel_labeled, \
        "Correct area should be smaller than Labeled"
    return pixel_correct/(pixel_labeled+1e-10), pixel_labeled  

# --------------------------------------
# old version (AP and pixel accuracy)
# --------------------------------------


def get_ap_scores(predict, target, ignore_index=-1):
    '''
    returns the average percision scores
    for prediction scores (1xCxHxW),  tensor on cpu, probability map
    and targets (1xHxW) with classes [0,1,2,3,...]
    '''
    total = []
    for pred, tgt in zip(predict, target.detach().cpu()):
        target_expand = tgt.unsqueeze(0).expand_as(pred) #CxHxW
        target_expand_numpy = target_expand.numpy().reshape(-1) # flatten 

        # Tensor process
        x = torch.zeros_like(target_expand) #CxHxW
        t = tgt.unsqueeze(0).clamp(min=0).long() #1xHxW
        target_1hot = x.scatter_(0, t, 1) #CxHxW

        predict_flat = pred.numpy().reshape(-1)
        target_flat = target_1hot.data.numpy().reshape(-1)

        p = predict_flat[target_expand_numpy != ignore_index]
        t = target_flat[target_expand_numpy != ignore_index]

        total.append(np.nan_to_num(average_precision_score(t, p)))
    return total
        
   

def pix_accuracy(predict, target):
    '''
    returns the the pixel-wise accuracy
    for prediction scores (1xCxHxW), tensor on cpu
    and targets (1xHxW) with classes [0,1,2,3,...]
    excluding the the bg class 0
    '''
 
    _, predict = torch.max(predict, 1)
    predict = predict.numpy()# + 1
    target = target.detach().cpu().numpy()# + 1
    pixel_labeled = np.sum(target > 0)
    pixel_correct = np.sum((predict == target) * (target > 0))
    assert pixel_correct <= pixel_labeled, \
        "Correct area should be smaller than Labeled"
    if pixel_labeled==0:
        print("This test image only contains bg")
    return pixel_correct/(pixel_labeled+1e-9), pixel_labeled


# --------------------------------------
# Functions for multi-label classification (match_ratio and classification accuracy)
# --------------------------------------



def calculate_match_number(outputs, labels):
    '''
    use for multilabel classification
    output: 
    '''
    outputs=outputs.detach().cpu() # N x num_classes
    labels=labels.detach().cpu() # N x num_classes

    predictions=(outputs>0).int()
    match_num=int(torch.all(predictions==labels,axis=1).sum())
    return match_num


def calculate_class_correct_num(outputs, labels):
    '''
    use for multilabel classification
    output: 
    '''
    outputs=outputs.detach().cpu() # N x num_classes
    labels=labels.detach().cpu() # N x num_classes

    predictions=(outputs>0).int()
    class_correct_num=(predictions==labels).sum(0) #num_clases

    # # sainty check
    # backup=torch.zeros(predictions.size(1))
    # for i in range(predictions.size(1)):
    #    backup[i]=(predictions[:,i]==labels[:,i]).sum()

    return class_correct_num



def clear_and_print(message):
    print(message)
    message=""
    return message

def gen_log_message(mode,epoch,loss,metrics,total_num_samples,return_value=False, print_class_iou=False):

    # summary info
    print("=============================="*4)
    message=f"====> Epoch {epoch}, {mode}-loss:{loss:.4f}, "
    if  "match_ratio" in metrics.keys():
        match_ratio= metrics["match_ratio"]['total_match_number']/total_num_samples
        message+=f'{mode}-accuracy: Match_ratio:{match_ratio*100:.2f} %.'
    else:
        match_ratio = 0
    message=clear_and_print(message)



    # classification info, shoulf be replace with F1 TODO
    if "class_acc" in metrics.keys():
        class_accuracy=metrics["class_acc"]['total_class_correct_num']/total_num_samples

        avg_class_acc=torch.mean(class_accuracy)
        for i in range(len(class_accuracy)):
            message+=f"C_{i+1}:{class_accuracy[i]:.2f}; "
        message=clear_and_print(f"{mode} : Classification accuracy: "+message+f"avg:{avg_class_acc:.2f}")
        print("------------------------------"*4)
    else:
        class_accuracy=None
        avg_class_acc=0
    

    # segmentation info
    if "iou" in metrics.keys():
        iou_class=metrics["iou"]['intersection_meter']/(metrics["iou"]['union_meter']+1e-8)
        mIoU=iou_class.mean()
        print(f"The {mode}_mIoU is {mIoU: .4f} and print_class_iou={print_class_iou}")
        
        if print_class_iou==True:
            for i in range(len(iou_class)):
                message+=f"C_{i+1}:{iou_class[i]:.2f}; "
            message=clear_and_print(f"{mode} : IoU class wise: "+message)
            print("------------------------------"*4)
    else:
        iou_class=None
        mIoU=0

        
    if "AP" in metrics.keys():
        ap= metrics["AP"]['ap_meter'].avg
        print(f"The ap value is {ap}")
    else:
        ap = 0

    if "pixel_acc" in metrics.keys():
        pixel_acc = metrics["pixel_acc"]['acc_meter'].avg.item()
        print(f"The pixel accuracy is {pixel_acc}")
    else:
        pixel_acc = 0

    if return_value:
        return mIoU,iou_class,match_ratio,avg_class_acc,class_accuracy, pixel_acc,ap
    else:
        return 

# --------------------------------------
# Functions for semantic segmentation
# --------------------------------------
# # metrics from kaggle
# def get_iou(outputs,labels):
#     num_classes=outputs.shape[1]
#     outputs = torch.stack([ torch.argmax(outputs,dim=1)==i for i in range(num_classes) ],dim=1)
#     labels = torch.stack([ labels==i for i in range(num_classes) ],dim=1)
#     intersection=torch.sum(torch.logical_and(outputs,labels), axis=(2,3))
#     union=torch.sum(torch.logical_or(outputs,labels), axis=(2,3))
#     iou = intersection  / (union + 1e-8) # n_samples x classes
#     iou = torch.sum(iou,dim=0) # classes
#     return iou

# metrics computing from https://github.com/Haochen-Wang409/U2PL/blob/ab3d2be313d4d6b2885e7a0213d4fce82a803b79/u2pl/utils/utils.py
def intersectionAndUnion(output, target, K, ignore_index=255):
    # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
    output=torch.argmax(output,dim=1).numpy()
    target=target.numpy()
    assert output.ndim in [1, 2, 3]
    assert output.shape == target.shape
    output = output.reshape(output.size).copy()
    target = target.reshape(target.size)
    output[np.where(target == ignore_index)[0]] = ignore_index # can be ignored as we set the boundary pixel to bg
    intersection = output[np.where(output == target)[0]]
    area_intersection, _ = np.histogram(intersection, bins=np.arange(K + 1))  
    area_output, _ = np.histogram(output, bins=np.arange(K + 1))
    area_target, _ = np.histogram(target, bins=np.arange(K + 1))
    area_union = area_output + area_target - area_intersection
    return area_intersection, area_union, area_target




# --------------------------------------
# Functions for early stopping
# --------------------------------------
class EarlyStopping():
    """
    Early stopping to stop the training when the loss does not improve after
    certain epochs.
    Code from https://debuggercafe.com/using-learning-rate-scheduler-and-early-stopping-with-pytorch/
    """
    def __init__(self, patience=20, min_delta=0):
        """
        :param patience: how many epochs to wait before stopping when loss is
               not improving
        :param min_delta: minimum difference between new loss and old loss for
               new loss to be considered as an improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss == None:
            self.best_loss = val_loss
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        elif self.best_loss - val_loss < self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


# --------------------------------------
# metrics
# --------------------------------------
def init_metrics(keywords_dict,num_classes):
    metrics={}
    if "classification" in keywords_dict.keys():
        tmp_list=keywords_dict["classification"]
        if "match_ratio" in tmp_list:
            metrics["match_ratio"]={}
            metrics["match_ratio"]['total_match_number']=0
        if "class_acc" in tmp_list:
            metrics["class_acc"]={}
            metrics["class_acc"]['total_class_correct_num']=torch.zeros(num_classes)
            # metrics["class_acc"]['total_class_num']=torch.zeros(num_classes)

    if "segmentation" in keywords_dict.keys():
        tmp_list=keywords_dict["segmentation"]
        if "iou" in tmp_list:
            metrics["iou"]={}
            metrics["iou"]['intersection_meter']= np.zeros(num_classes)
            metrics["iou"]['union_meter']= np.zeros(num_classes)
        if "AP" in tmp_list:
            metrics["AP"]={}
            metrics["AP"]['ap_meter']=AverageMeter()
        if "pixel_acc" in tmp_list:
            metrics["pixel_acc"]={}
            metrics["pixel_acc"]['acc_meter']=AverageMeter()
    
    return metrics

def update_metrics(metrics,outputs,labels,pred_maps,mask_scores,sem_gts ):
    num_classes=labels.shape[1]
    outputs=outputs.detach().cpu()
    labels=labels.detach().cpu()
    pred_maps # will be connected to adjusted_heatmaps
    sem_gts=sem_gts.detach().cpu()

    # "classification"
    if "match_ratio" in metrics.keys():
        num_match=calculate_match_number(outputs,labels)
        metrics["match_ratio"]['total_match_number']+= num_match

    if "class_acc" in metrics.keys():
        class_correct_num=calculate_class_correct_num(outputs,labels)
        metrics["class_acc"]['total_class_correct_num']+= class_correct_num
        # metrics["class_acc"]['total_class_num']+= labels.sum(0) 

    #"segmentation" :
    
    if "iou" in metrics.keys():
        intersection, union, _=intersectionAndUnion(pred_maps, sem_gts, num_classes, 255)
        metrics["iou"]['intersection_meter'] += intersection
        metrics["iou"]['union_meter'] += union


    if "AP" in metrics.keys():
        # mask_scores=F.softmax(pred_maps, 1)
        ap_score,num_data=calculate_ap_scores(mask_scores, sem_gts)
        metrics["AP"]['ap_meter'].update(ap_score,num_data)

    if "pixel_acc" in metrics.keys():
        # mask_scores=F.softmax(pred_maps, 1)
        pixel_acc, num_pixels=calculate_pix_accuracy(mask_scores, sem_gts)
        metrics["pixel_acc"]['acc_meter'].update(pixel_acc, num_pixels)

    return metrics


        