### This file is a utils file dedicated for all functions needed for weights and biases and logging
import wandb
import numpy as np
import torch
import matplotlib.pyplot as plt
from .plot_utils import color_heatmap
import PIL
import sys
def fixed_random_examples(data_length,n_examples,seed=0):
    
    np.random.seed(seed)
    return np.random.choice(data_length,n_examples,replace=False)


class watch_gradients(object):
    def __init__(self,init_calls = 0):
        self.calls = init_calls

    def update(self,model,logging_period = 10, iter = None):
        '''
        the gradients of the model will be logged every (logging_period) call
        with a defined iteration or the internal wandb tracked one
        '''
        if self.calls % logging_period == 0:
            for name, param in model.named_parameters():
                wandb.log({'gradients/'+'{}_grad'.format(name): param.grad},step=iter)
        
        self.calls += 1


def log_semantic_segmentation(img,gt_mask,pred,class_dict,iter_idx,gt_class=None,pred_class=None,title="examples"):
    '''
    method for logging one torch images (C,H,W)
    including the semantic gt (H,W)
    and the prediction (classes,H,W)
    '''

    # adjust if no label is assigned
    gt_mask[gt_mask == -1] = 42

    if gt_class == None:
        log_img = wandb.Image(img.permute((1,2,0)).cpu().numpy(),masks={
            "ground_truth":{"mask_data": gt_mask.cpu().numpy(),"class_labels":class_dict},    
            "prediction":{"mask_data": torch.argmax(pred,dim=0).cpu().numpy(),"class_labels":class_dict},                     
            })
    else:
        # construct easy readable pred gt comparison for bottleneck
        G = gt_class.cpu().numpy()
        P = np.round(torch.sigmoid(pred_class).cpu().numpy(),decimals=2)
        output = ''
        for i in range(len(gt_class)):
            output += str(class_dict[i])+': ('+str(G[i])+','+ str(P[i])+') \n'

        log_img = wandb.Image(img.permute((1,2,0)).cpu().numpy(),masks={
        "ground_truth":{"mask_data": gt_mask.cpu().numpy(),"class_labels":class_dict},    
        "prediction":{"mask_data": torch.argmax(pred,dim=0).cpu().numpy(),"class_labels":class_dict},                    
        },caption=output)
    wandb.log({title:log_img},step=iter_idx)




def log_heatmaps(heatmaps,class_dict,iter_idx,title="heatmap"):
    '''
    logs the heatmaps (C,H,W, torch tensor) produced
    '''
    # TODO map color positiv and negative values
    hmin = torch.amin(heatmaps, dim=(1,2))
    hmax = torch.amax(heatmaps, dim=(1,2))

    # asymmetric scaling preserving zeros
    heatmaps[heatmaps>0]=heatmaps[heatmaps>0]/torch.max(heatmaps)
    heatmaps[heatmaps<0]=heatmaps[heatmaps<0]/torch.abs(torch.min(heatmaps))

    for c in range(heatmaps.size(0)):
        class_title = title+'_'+class_dict[c]
        # print(f"for class {c} and class_title {class_title}")

        heatmap_color=color_heatmap(heatmaps[c].cpu().numpy())


        caption=class_title+' min '+str(float(hmin[c]))+' max '+str(float(hmax[c]))
        log_map = wandb.Image((heatmap_color*255).astype(np.uint8),caption=caption)
        # log_map = wandb.Image(PIL.Image.fromarray(heatmap,mode="F"),caption=class_title)
        wandb.log({class_title:log_map},step=iter_idx)





# def color_map(img,positiv_color=torch.tensor([255,0,0]),negativ_color=torch.tensor([0,0,255]),bg=torch.tensor([255,255,255])):
#     '''
#     converts image (H,W) with range [-1,1] to color map

#     return: cmap: colored heatmap, torch.tensor, shape [H,W,3]
#     '''
#     # TODO need to check this part of code
#     cmap = img.cpu().repeat(3,1,1).permute((1,2,0)) # HxWx3
#     p= torch.clamp(cmap, min=0)
#     n= torch.clamp(-cmap, min=0)
#     cmap= p * positiv_color+ (1-p) * (cmap>=0) * bg + n * negativ_color+ (1-n) * (cmap<0) * bg
#     return cmap
    




def log_image(img,iter_idx,class_dict=None,gt_class=None,pred_class=None,title="examples"):
    '''
    method for logging one torch images (C,H,W)
    '''

    if gt_class == None:
        log_img = wandb.Image(img.permute((1,2,0)).numpy())
    else:
        # construct easy readable pred gt comparison for bottleneck
        G = gt_class.numpy()
        P = np.round(torch.sigmoid(pred_class).numpy(),decimals=2)
        output = ''
        for i in range(len(gt_class)):
            output += str(class_dict[i])+': ('+str(G[i])+','+ str(P[i])+') \n'

        log_img = wandb.Image(img.permute((1,2,0)).numpy(),caption=output)
    
    wandb.log({title:log_img},step=iter_idx)




def log_training(iter,epoch,idx_batch,len_batch,
                        bottleneck_loss,heatmap_loss,metrics,
                        input_imgs,sem_gts,heatmaps,mask_scores,class_dict,
                        gt_class,pred_class,log_imgs=True, media_log_interval = 500):
    '''
    mask scores: torch.tensor, cpu, shape N x 21 x H x W, probability of each pixels to be a certain class
                range (0,1) or {0, 1} (for the case of sigmoid_per_predicted_class)?
    '''
    #iter_idx = ((epoch-1)*batches_in_epoch+batch)*len_batch
    wandb.log({'epoch':epoch-1},step=iter)
    
    # log losses
    wandb.log({'bottleneck_loss': bottleneck_loss/len_batch},step=iter)
    if np.isnan(heatmap_loss) == False: # workaround for semi-supervised setup
        wandb.log({'heatmap_loss': heatmap_loss/len_batch},step=iter)
    
    # log seen images
    if log_imgs:
        for idx_img in range(len_batch):
            if ((idx_img + iter) % media_log_interval) == 0 or (idx_batch == 0 and idx_img == 0):
                input_imgs=input_imgs.detach().cpu()
                sem_gts=sem_gts.detach().cpu()
                gt_class=gt_class.detach().cpu()
                pred_class=pred_class.detach().cpu()
                heatmaps=heatmaps.detach().cpu()

                log_semantic_segmentation(input_imgs[idx_img],sem_gts[idx_img],mask_scores[idx_img],class_dict,idx_img + iter,gt_class[idx_img],pred_class[idx_img],"Training examples")
                log_heatmaps(heatmaps[idx_img],class_dict,idx_img + iter)






class storing_imgs(object):
    """
    for storing torch images (C,H,W) -> (N,C,H,W)
    including the semantic gt (H,W) -> (N,H,W)
    and the prediction (classes,H,W) -> (N,classes,H,W)
    """
    def __init__(self,n_batches,test_batch_size, n_examples = 25,seed=0):
        self.initialized = False
        self.imgs = None
        self.gt_mask = None
        self.pred_mask = None
        self.batches_to_track = fixed_random_examples(n_batches,n_examples,seed=seed)
        print(f"The test imge idx (N_record_test = {len(self.batches_to_track)}) falls in this list {self.batches_to_track} will be tracked.")
        self.batch_size=test_batch_size
        # when initialize storing imgs, the idx_img=0
        self.idx_img=0

    def get_right_format(self):
        self.img = self.batch_imgs[self.img_of_batch].unsqueeze(0) # (1,C,H,W)
        self.gt_mask = self.batch_gts[self.img_of_batch].squeeze().unsqueeze(0) # (1,H,W)
        self.pred_mask = self.batch_preds[self.img_of_batch].unsqueeze(0) # (B,classes,H,W) -> (1,classes,H,W)

    def initialize(self):
        self.get_right_format()
        self.imgs = self.img
        self.gt_masks = self.gt_mask
        self.pred_masks = self.pred_mask
        self.initialized = True

    def update(self, idx_batch, batch_imgs,batch_gts,batch_preds):
        self.batch_imgs = batch_imgs
        self.batch_gts = batch_gts
        self.batch_preds = batch_preds
        
        for self.img_of_batch in range(self.batch_gts.shape[0]):
#             idx_img=idx_batch*len_batch+self.img_of_batch #bug as last image in the last batch has same idx img as the first image in the next batch
            # update idx_img
            self.idx_img+=1
            # assert self.idx_img==idx_batch*self.batch_size+self.img_of_batch+1,f"{self.idx_img} should be equal to {idx_batch*self.batch_size+self.img_of_batch+1}"
            if self.idx_img in self.batches_to_track:
                if not self.initialized:
                    self.initialize()
                else:
                    self.append()

    def append(self):
        self.get_right_format()
        # self.imgs = torch.concat((self.imgs,self.img))
        # self.gt_masks = torch.concat((self.gt_masks,self.gt_mask))
        # self.pred_masks = torch.concat((self.pred_masks,self.pred_mask))
        self.imgs = torch.cat((self.imgs,self.img))
        self.gt_masks = torch.cat((self.gt_masks,self.gt_mask))
        self.pred_masks = torch.cat((self.pred_masks,self.pred_mask))

    def get_imgs(self):
        return self.imgs

    def get_gt_masks(self):
        return self.gt_masks

    def get_pred_masks(self):
        return self.pred_masks






# def storing_imgs(batch,log_batch,
#                 input_imgs,sem_gts,heatmaps,
#                 log_examples,log_gt_heatmaps,log_pred_heatmaps,
#                 img_of_batch = 0):
#     '''
#     method for storing torch images (C,H,W) -> (N,C,H,W)
#     including the semantic gt (H,W) -> (N,H,W)
#     and the prediction (classes,H,W) -> (N,classes,H,W)
#     '''
    
#     if batch in log_batch:
#         example = input_imgs[img_of_batch].unsqueeze(0) # (1,C,H,W)
#         gt_heatmap = sem_gts[img_of_batch].squeeze().unsqueeze(0) # (1,H,W)
#         pred_heatmap = heatmaps[img_of_batch].unsqueeze(0) # (B,classes,H,W) -> (1,classes,H,W)
#         if torch.is_tensor(log_examples):
#             log_examples = torch.concat((log_examples,example))
#             log_gt_heatmaps = torch.concat((log_gt_heatmaps,gt_heatmap))
#             log_pred_heatmaps = torch.concat((log_pred_heatmaps,pred_heatmap))
#         else:
#             log_examples = example
#             log_gt_heatmaps = gt_heatmap
#             log_pred_heatmaps = pred_heatmap
            
#     return log_examples,log_gt_heatmaps,log_pred_heatmaps




def log_testing(iter_idx,
                test_loss,mIoU,match_ratio,
                avg_seg_pxl_accu,avg_seg_ap,
                imgs,sem_gts,heatmaps,class_dict,
                columns = 5):
            
    # 1) log average loss
    wandb.log({'test_loss': test_loss},step=iter_idx)

    # 2) log metrics
    wandb.log({'test_mIoU':mIoU},step=iter_idx)
    wandb.log({'test_match_ratio':match_ratio},step=iter_idx)  
    wandb.log({'test_seg_pixel_accuracy':avg_seg_pxl_accu},step=iter_idx) 
    wandb.log({'test_seg_AP':avg_seg_ap},step=iter_idx) 

    # 3) display test examples in a grid like structure
    rows = int(np.ceil(imgs.size(0)/columns))
    c = imgs.size(1)
    h = imgs.size(2)
    w = imgs.size(3)
    classes = heatmaps.size(1)
    grid_imgs = torch.zeros((c,rows*h,columns*w),dtype=imgs.dtype)
    grid_gt = torch.zeros((rows*h,columns*w),dtype=sem_gts.dtype)
    grid_heatmaps = torch.zeros((classes,rows*h,columns*w),dtype=heatmaps.dtype)

    for i in range(imgs.size(0)):
        column = i % columns
        row = int(i/columns)

        grid_imgs[:,row*h:(row+1)*h,column*w:(column+1)*w] = imgs[i,:]
        grid_gt[row*h:(row+1)*h,column*w:(column+1)*w] = sem_gts[i]
        grid_heatmaps[:,row*h:(row+1)*h,column*w:(column+1)*w] = heatmaps[i,:]

    log_semantic_segmentation(grid_imgs,grid_gt,grid_heatmaps,class_dict,iter_idx,title="Testing examples")
    # TODO incoorperate class specific metrics
        

def log_testing_per_class_metrics(iter_idx,tabele_title, wandb_table, per_class_metrics,count,log_times=17):
    wandb_table.add_data(iter_idx, *per_class_metrics)  
    if count>=log_times:
        wandb.log({tabele_title:wandb_table}) 

    




def log_pretraining_train(iter,epoch,idx_batch,len_batch,
                        bottleneck_loss,input_imgs,
                        class_dict,gt_class,pred_class,media_log_interval = 500):
    
    #iter_idx = ((epoch-1)*batches_in_epoch+batch)*len_batch
    wandb.log({'epoch':epoch-1},step=iter)
    
    # log losses
    wandb.log({'pretraining_bottleneck_loss': bottleneck_loss/len_batch},step=iter)
    
    # log seen images (just always take first of batch) and outputs an gt

    for idx_img in range(len_batch):
        if (((idx_img + iter) % media_log_interval) == 0) or (idx_batch == 0 and idx_img == 0):
            input_imgs=input_imgs.detach().cpu()
            gt_class=gt_class.detach().cpu()
            pred_class=pred_class.detach().cpu()
            log_image(input_imgs[idx_img],iter,class_dict,gt_class[idx_img],pred_class[idx_img],"Pretraining examples")




def log_pretraining_test(iter_idx,test_loss,match_ratio,avg_classification_accu):
            
    # 1) log average loss
    wandb.log({'pretraining_test_loss': test_loss},step=iter_idx)

    # 2) log metrics
    wandb.log({'pretraining_test_match_ratio':match_ratio},step=iter_idx)  
    
    wandb.log({'pretraining_test_avg_classification_accu':avg_classification_accu},step=iter_idx)  
    
