from __future__ import print_function
import torch
from torch import nn, optim
from torch.nn import functional as F
import sys
sys.path.append('../')

from datasets import PASCAL
import numpy as  np

import os
from utils.plot_utils import *
from utils.metrics_utils import *
from utils.sampling_utils import *
# import torch.utils.data.sampler as sampler

# set up wandb 
import utils.wandb_utils as wandb_utils
from network import create_model
from collections import OrderedDict
from utils.loss_utils import compute_supervised_classification_loss
# --------------------------------------
# Main function for training and test
# --------------------------------------
class Baseline(object):
    '''
    This is implementation for multi-task UNet and UNet' training and testing.

    '''
    def __init__(self, args, logger):
        self.args = args
        self.device = torch.device("cuda" if args.cuda else "cpu")
        self._init_dataset()

        self.seen_train_imgs = 0
        self.max_mIoU = 0 # for model saving
        self.max_mIoU_iter= 0 # for model saving

     
        if args.add_classification==False:
            #full supervision mode
            print(f"The model is trained in FULL supervised manner with {len(self.data.training_dataset.idx_mask)} images")
            args.batch_size=min(len(self.data.training_dataset.idx_mask),args.batch_size)
            self.train_loader=torch.utils.data.DataLoader(self.data.training_dataset,batch_size=args.batch_size, shuffle=True)
            self.test_loader=torch.utils.data.DataLoader(self.data.testing_dataset, batch_size=40)
 
        else:
            #semi supervision mode
            print(f"The model is trained in SEMI supervised manner with {len(self.data.training_dataset.idx_mask)} images")
            self.train_sampler = semi_supervised_sampler(self.data.training_dataset.idx_mask, self.data.training_dataset.idx_no_mask,
                                                        args.batch_size, len(self.data.training_dataset),0.5, 1.0,args.seed)
            self.train_loader=torch.utils.data.DataLoader(self.data.training_dataset,sampler=self.train_sampler.sample(),batch_size=args.batch_size)
            self.test_loader=torch.utils.data.DataLoader(self.data.testing_dataset, batch_size=40)
        if args.pre_batch_size>0 and args.pre_epochs>0:
            self.pre_train_loader=torch.utils.data.DataLoader(self.data.pre_training_dataset,batch_size=args.pre_batch_size,shuffle=True)
            self.pre_test_loader=torch.utils.data.DataLoader(self.data.pre_testing_dataset, batch_size=args.pre_batch_size)


        # model
        assert args.model in ["std_unet","mt_unet"], "The loaded model should be std_unet or mt_unet."
        if args.model=="std_unet":
            self.model = create_model(
                    "Unet",
                    encoder_name=args.encoder,
                    encoder_weights="imagenet",
                    in_channels=3,
                    classes=args.num_classes,
                    decoder_symmetric=True,
                    add_classification=False
                ).to(self.device)
        elif args.model== "mt_unet":
            self.model = create_model(
                    "Unet",
                    encoder_name=args.encoder,
                    encoder_weights="imagenet",
                    in_channels=3,
                    classes=args.num_classes,
                    decoder_symmetric=True,
                    add_classification=True
                ).to(self.device)
        else:
            raise NotImplementedError

        self.model.to(self.device)


        self.optimizer = optim.AdamW(self.model.parameters(), lr=args.lr)
        self.alpha = args.loss_impact_seg #hyperparam for heatmap loss
        self.beta = args.loss_impact_bottleneck #hyperparam for classification loss

        self.logger = logger

        self.train_metrics={"classification": ["match_ratio","class_acc"],
                        "segmentation":["iou"]
            }
        self.test_metrics={"classification": ["match_ratio","class_acc"],
                        "segmentation":["iou"]
            }
  
      

    def _init_dataset(self):
        if self.args.dataset == 'PASCAL':
            self.data = PASCAL(self.args)
            self.img_channel=3
            self.input_size=(3,256,256) 
        else:
            print("Dataset not supported")

  

    def save_model(self,epoch,keyword):
        torch.save(self.model.state_dict(),f"snapshot/{self.args.model}_{epoch}_{keyword}_{self.args.num_classes}.pth")



    def loss_function(self, pred_class_scores, class_gt, heatmaps,seg_gt,seg_gt_exists):

        hm_sf=1.0 # heatmap scale factor 
        hm_offset=0.0 # heatmap offset 
        heatmap_loss=nn.CrossEntropyLoss(reduction="mean",ignore_index=-1)(input=hm_sf*(heatmaps+hm_offset),target=seg_gt) 
      

        if self.args.add_classification:
            classification_loss=compute_supervised_classification_loss(pred_class_scores,class_gt.float(),self.device)*self.args.batch_size
        else:
            classification_loss = torch.tensor(0).to(self.device)

        return self.beta*classification_loss+self.alpha*heatmap_loss,  self.alpha*heatmap_loss , self.beta*classification_loss

    
   

    def inferring_mask_scores(self,class_pred,heatmap_pred,class_thresh=0.5):
        '''
        inferring mask scores (NxCxHxW) with values 0,..,1
        from heatmaps (NxCxHxW) [-infty,infty]
        based on the prediction of class
        '''
        # get predicted classes
        binary_pred = torch.sigmoid(class_pred.detach().cpu()).gt(class_thresh).long()

        # infer mask
        adjusted_heatmap = heatmap_pred.detach().cpu().clone()
        mask_pred = F.softmax(adjusted_heatmap, 1)
        
        return mask_pred,binary_pred,adjusted_heatmap


    # --------------------------------------
    # Functions for pretrain
    # --------------------------------------
    def pretraining_train(self,epoch):
        assert self.args.model=='mt_unet','Only multi-task UNet (mu_unet) can do pretraining.'

        self.model.train()
        train_loss = 0
        # class_counts = torch.zeros(self.args.num_classes).to(self.device) # TODO put into dedicated function - push to init
        metrics=init_metrics(self.pretraining_train_metrics,self.args.num_classes)
        for batch_idx, (data, class_labels,sem_gts,_) in enumerate(self.pre_train_loader):  
            
           
            data, class_labels= data.to(self.device), class_labels.to(self.device)       
            self.optimizer.zero_grad()
            _,class_scores= self.model(data)
            loss = compute_supervised_classification_loss(class_scores,class_labels,self.device)
            loss.backward()
            train_loss += loss.item()
            self.optimizer.step()

            with torch.no_grad():
                metrics=update_metrics(metrics,class_scores,class_labels,None,None,sem_gts)
                if self.args.wandb != 'None':
                    wandb_utils.log_pretraining_train(self.seen_train_imgs,epoch,batch_idx,len(data),loss.item(),data,
                        self.data.pre_training_dataset.new_classes,class_labels,class_scores)
                    
            self.seen_train_imgs += len(data)
                
        train_loss/= len(self.pre_train_loader.dataset)
        gen_log_message("pre-train",epoch,train_loss , metrics,len(self.pre_train_loader.dataset),print_class_iou=False)
        


    def pretraining_test(self,epoch):
        assert self.args.model=='mt_unet','Only multi-task UNet (mu_unet) can do pretesting.'

        self.model.eval()
        test_loss = 0

        metrics=init_metrics(self.pretraining_test_metrics,self.args.num_classes)
        with torch.no_grad():
            for i, (data, class_labels,sem_gts,_) in enumerate(self.pre_test_loader):
       
                data, class_labels= data.to(self.device), class_labels.to(self.device)
                _,class_scores= self.model(data)
                loss = compute_supervised_classification_loss(class_scores,class_labels,self.device)
                test_loss += loss.item()
                metrics=update_metrics(metrics,class_scores,class_labels,None,None,sem_gts)
           
        test_loss /= len(self.pre_test_loader.dataset)

        _,_,_,_,match_ratio,avg_class_acc,_,_,_ = gen_log_message("pre-test",epoch,test_loss, metrics,len(self.pre_test_loader.dataset),return_value=True,print_class_iou=False)
       

        if self.args.wandb != 'None':
            wandb_utils.log_pretraining_test(self.seen_train_imgs,test_loss,match_ratio,avg_class_acc)  

        self.save_model(epoch,"pre_train") 




    # --------------------------------------
    # Functions for offical train
    # --------------------------------------
        
    def train(self, epoch):
        self.model.train()
        train_loss = 0
        metrics=init_metrics(self.train_metrics,self.args.num_classes)
        for batch_idx, (data, class_labels,sem_gts,sem_gt_exist) in enumerate(self.train_loader):
            # 1) prepare data
            sem_gts=sem_gts.long().to(self.device)
            class_labels=class_labels.to(self.device)
            data = data.to(self.device)

            self.optimizer.zero_grad()
    
            if self.args.add_classification==False:
                heatmaps = self.model(data) 
                class_scores=torch.mean(heatmaps.data, dim=(2,3)) 
            else:
                heatmaps,class_scores = self.model(data)

            loss, loss1, loss2= self.loss_function(class_scores,class_labels,heatmaps,sem_gts,sem_gt_exist)
            loss.backward()
            
            train_loss += loss.item()        
            self.optimizer.step()
           
            # 3) detach all computational graphs for further postprocessing
            with torch.no_grad():
                
                # 4) get masks scores based on predicted classes
                mask_scores,_,adjusted_heatmap = self.inferring_mask_scores(class_scores,heatmaps)
                # calculate the metrics and update
                metrics=update_metrics(metrics,class_scores,class_labels,adjusted_heatmap,mask_scores,sem_gts)
            
                if self.args.wandb != 'None':
                    wandb_utils.log_training(self.seen_train_imgs,epoch,batch_idx,len(data),
                        loss2.item(),loss1.item(),metrics,
                        data,sem_gts,heatmaps,mask_scores,self.data.training_dataset.new_classes,
                        class_labels,class_scores,self.args.log_imgs,media_log_interval=3000)  #todo metrics is not used in wandb_utils.log_training
         
                self.seen_train_imgs += len(data)
                if self.args.add_classification==False:
                    remain=self.seen_train_imgs //1500
                    if self.seen_train_imgs<remain*1500+self.args.batch_size:
                        self.test(epoch)
                else:
                    remain=self.seen_train_imgs //3000
                    if self.seen_train_imgs<remain*3000+self.args.batch_size:
                        self.test(epoch)


             
        train_loss=train_loss/ len(self.train_loader.dataset)
        gen_log_message("train",epoch,train_loss , metrics,len(self.train_loader.dataset),print_class_iou=False)

        
    def test(self, epoch):
        
        self.model.eval()
        metrics=init_metrics(self.test_metrics,self.args.num_classes)
        test_loss = 0
        
        storing_examples = wandb_utils.storing_imgs(n_batches = len(self.data.testing_dataset),
                                                    test_batch_size=self.args.batch_size,seed=0)
       

        with torch.no_grad():
            for batch_idx, (data, class_labels,sem_gts,sem_gt_exists) in enumerate(self.test_loader):
               
                sem_gts=sem_gts.long().to(self.device)
                data = data.to(self.device)
                class_labels=class_labels.to(self.device)

                if self.args.add_classification==False:
                    heatmaps = self.model(data) 
                    class_scores=torch.mean(heatmaps.data, dim=(2,3)) 
                else:
                    heatmaps,class_scores = self.model(data)
               
                tloss,_,_ = self.loss_function(class_scores,class_labels,heatmaps,sem_gts,sem_gt_exists)
                test_loss += tloss.item()


                # get masks scores based on predicted classes
                mask_scores,_,adjusted_heatmap = self.inferring_mask_scores(class_scores,heatmaps)

                # calculate the metrics and update# compute metrics 
                metrics=update_metrics(metrics,class_scores,class_labels,adjusted_heatmap,mask_scores,sem_gts)
            
                # save images and heatmaps for logging
                if self.args.wandb != 'None':
                    storing_examples.update(batch_idx, data.detach().cpu(),sem_gts.detach().cpu(),mask_scores)  
        
        test_loss /= len(self.test_loader.dataset)
        mIoU,_,match_ratio,avg_class_acc,_,pixel_acc,ap = gen_log_message("test",epoch,test_loss, metrics,len(self.test_loader.dataset),return_value=True,print_class_iou=True)
     
        
        if self.args.wandb != 'None':
            wandb_utils.log_testing(self.seen_train_imgs,
                                    test_loss,mIoU,match_ratio,pixel_acc,ap,
                                    storing_examples.get_imgs(),storing_examples.get_gt_masks(),storing_examples.get_pred_masks(),self.data.training_dataset.new_classes)  

       
        if mIoU >= self.max_mIoU:
            # first save the new checkpoint then update self.max_mIoU, self.max_mIoU_iter
            if self.args.save_folder is not None:
                # save the model
                os.makedirs(f"{self.args.save_folder}/{self.args.encoder}_seed{self.args.seed}",exist_ok=True)
                torch.save(self.model.state_dict(), 
                    f"{self.args.save_folder}/{self.args.encoder}_seed{self.args.seed}"+
                        f"/{self.args.model}_{self.args.encoder}_iter{self.seen_train_imgs}_lab{self.args.num_labels}_lr{self.args.lr}_bs{self.args.batch_size}.pth")
                # update mIoU
                try:
                    os.remove(f"{self.args.save_folder}/{self.args.encoder}_seed{self.args.seed}"+
                            f"/{self.args.model}_{self.args.encoder}_iter{self.max_mIoU_iter}_lab{self.args.num_labels}_lr{self.args.lr}_bs{self.args.batch_size}.pth")
                except FileNotFoundError:
                    pass
            self.max_mIoU= mIoU
            self.max_mIoU_iter=self.seen_train_imgs


        self.logger.info(
        f'EPOCH: {epoch:04d} ITER: {self.seen_train_imgs:04d} | Test [Loss | mIoU | Acc.]: {test_loss :.4f} {mIoU :.4f} {pixel_acc:.4f}')
        self.logger.info(f'Top: mIoU {self.max_mIoU :.4f}')
        
        self.model.train()


    # --------------------------------------
    # Functions for loading pretrain weights
    # --------------------------------------
        
    def align_checkpoint_dict(self, pretrain_weight_name):
        checkpoint =torch.load(pretrain_weight_name + '.pth')
        # standardize the checkpoint keys
        cl_head_checkpoint=OrderedDict()      
        # filter out the decoder layers checkpoint
        for key in list(checkpoint.keys()):
            if 'tied_weights_decoder' in key: #or 'classifier' in key:
                checkpoint[key.replace('tied_weights_decoder.', 'tied_weights_decoder.decoder.')] = checkpoint[key]
                del checkpoint[key]
            elif 'encoder.backbone.' in key:
                
                checkpoint[key.replace('encoder.backbone.', 'encoder.')] = checkpoint[key]
                del checkpoint[key]


        return checkpoint
    
    def load_pretrain_model_variant(self,pretrain_weight_name):
        print("Only load the encoder pretrain-weight part")
        # only pick the encoder pretrain weight
        checkpoint =torch.load(pretrain_weight_name + '.pth')
        # print(self.model.encoder)
        # print(checkpoint.keys())
        # filter out the decoder layers checkpoint
        for key in list(checkpoint.keys()):
            if 'tied_weights_decoder' in key: #or 'classifier' in key:
                del checkpoint[key]
            elif 'encoder.backbone.' in key:            
                checkpoint[key.replace('encoder.backbone.', '')] = checkpoint[key]
                del checkpoint[key]

        self.model.encoder.load_state_dict(checkpoint)
   



    def delete_decoder_weights_checkpoint(self, checkpoint):
        for key in list(checkpoint.keys()):
            if 'tied_weights_decoder' in key: 
                del checkpoint[key]
            elif 'encoder.' in key: 
                checkpoint[key.replace('encoder.', '')] = checkpoint[key]
                del checkpoint[key]
        return checkpoint

    # This is the load_pretrain_model function for unrolled_lrp model, not applicable to unet model
    def load_pretrain_model(self,pretrain_weight_name):
        print(f"Load pretrained weights {pretrain_weight_name + '.pth'}")
        checkpoint =torch.load(pretrain_weight_name + '.pth')
        cl_head_checkpoint=OrderedDict() 
        for key in list(checkpoint.keys()):
            if 'tied_weights_decoder' in key: #or 'classifier' in key:
                del checkpoint[key]
            elif 'encoder.backbone.' in key:
                
                checkpoint[key.replace('encoder.backbone.', '')] = checkpoint[key]
                del checkpoint[key]

        # fill in the classification head checkpoint
        for key in list(checkpoint.keys()):
            if 'classifier' in key  or 'fc' in key:
                cl_head_checkpoint[key]=checkpoint[key]
                del checkpoint[key]

        if self.args.model=='std_unet':
            self.model.encoder.load_state_dict(checkpoint)
        elif self.args.model=='mt_unet':
            self.model.encoder.load_state_dict(checkpoint)
            self.model.classification_head.load_state_dict(cl_head_checkpoint)
        else:
            raise NotImplementedError
        
        self.load_pretrain_model_variant(pretrain_weight_name)

        return 
        