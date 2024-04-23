from __future__ import print_function
import torch
from torch import nn, optim
from torch.nn import functional as F
from collections import OrderedDict
import sys
sys.path.append('../')

import logging
from datasets import  PASCAL
import numpy as  np
import os
from utils.plot_utils import *
from utils.metrics_utils import *
from utils.sampling_utils import *
import utils.wandb_utils as wandb_utils
from network import create_model
from utils.loss_utils import compute_supervised_classification_loss
# --------------------------------------
# Main function for training and test
# --------------------------------------
class TW_Autoencoder(object):
    def __init__(self, args, logger):
        self.args = args
        self.device = torch.device("cuda" if args.cuda else "cpu")
        self._init_dataset()

        # setup universal counter for seen training imgs
        self.seen_train_imgs = 0
        self.max_mIoU = 0 # for model saving
        self.max_mIoU_iter= 0 # for model saving

        #semi supervision mode
        print(f"The model is trained in SEMI supervised manner with {len(self.data.training_dataset.idx_mask)} images")
        self.train_sampler = semi_supervised_sampler(self.data.training_dataset.idx_mask, self.data.training_dataset.idx_no_mask,
                                                    args.batch_size, len(self.data.training_dataset),0.5, 1.0,args.seed)
        self.train_loader=torch.utils.data.DataLoader(self.data.training_dataset,sampler=self.train_sampler.sample(),batch_size=args.batch_size)
        self.test_loader=torch.utils.data.DataLoader(self.data.testing_dataset, batch_size=args.batch_size)
        if args.pre_batch_size>0 and args.pre_epochs>0:
            self.pre_train_loader=torch.utils.data.DataLoader(self.data.pre_training_dataset,batch_size=args.pre_batch_size,shuffle=True)
            self.pre_test_loader=torch.utils.data.DataLoader(self.data.pre_testing_dataset, batch_size=args.pre_batch_size)

        # model
        assert args.model == "unrolled_lrp", "The loaded model should be unrolled_lrp."
        self.model = create_model(
            "Unrolled_lrp",
            encoder_name=args.encoder,
            encoder_weights="imagenet",
            in_channels=3,
            classes=args.num_classes,
            xai=args.xai,
            epsilon=args.epsilon,
            alpha=args.alpha,
            detach_bias=True,
            # input_size is used for running test_forward for unrolled_lrp model
            # And it will help to predefine the output_padding for ConvTranspose2d layer and kernel size of the inv_adaptive avgpool layer in the decoder
            # here we only consider the case that the eval/test images size is same as the train images, could be improved to be general later
            input_size=(args.crop_size,args.crop_size),
            ablation_test=args.ablation_test,
            normal_relu=args.normal_relu,
            normal_deconv=args.normal_deconv,
            normal_unpool=args.normal_unpool,
            multiply_input=args.multiply_input,
            remove_heaviside=args.remove_heaviside,
            remove_last_relu=args.remove_last_relu,
            add_bottle_conv=args.add_bottle_conv
        ).to(self.device)


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

        # self.pretraining_train_metrics={"classification": ["match_ratio","class_acc"]}
        # self.pretraining_test_metrics={"classification": ["match_ratio","class_acc"]}

    def _init_dataset(self):
        if self.args.dataset == 'PASCAL':
            self.data = PASCAL(self.args)
            self.img_channel=3
            self.input_size=(3,256,256) 
        else:
            print("Dataset not supported")

   


    def save_model(self,epoch,keyword):
        torch.save(self.model.state_dict(),f"snapshot/{self.args.encoder}_{epoch}_{keyword}_{self.args.num_classes}.pth")


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

        self.model.train()
        train_loss = 0
        # class_counts = torch.zeros(self.args.num_classes).to(self.device) # TODO put into dedicated function - push to init
        metrics=init_metrics(self.pretraining_train_metrics,self.args.num_classes)
        for batch_idx, (data, class_labels,sem_gts,_) in enumerate(self.pre_train_loader):  
            
           
            data, class_labels= data.to(self.device), class_labels.to(self.device)       
            self.optimizer.zero_grad()
            class_scores= self.model(data,class_labels, only_classification=True)
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

        self.model.eval()
        test_loss = 0

        metrics=init_metrics(self.pretraining_test_metrics,self.args.num_classes)
        with torch.no_grad():
            for i, (data, class_labels,sem_gts,_) in enumerate(self.pre_test_loader):
       
                data, class_labels= data.to(self.device), class_labels.to(self.device)
                class_scores= self.model(data,class_labels, only_classification=True)
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
        watch = wandb_utils.watch_gradients()

        # reshuffle sampler again and setup again trainloader
        self.train_loader=torch.utils.data.DataLoader(self.data.training_dataset,sampler=self.train_sampler.sample(),batch_size=self.args.batch_size)

        for batch_idx, (data, class_labels,sem_gts,sem_gt_exist) in enumerate(self.train_loader): 
            
            # measure number of samples before it goes to if function about self.args.only_send_labeled_data
            num_samples=len(data)
            # 1) prepare data
            data, class_labels, sem_gts = data.to(self.device),class_labels.to(self.device), sem_gts.long().to(self.device)


            
            # 2) apply model on data and update gradients 
            self.optimizer.zero_grad()

            if self.args.iterative_gradients:
                class_scores = self.model(data,class_labels, only_classification=True)
                heatmaps_detached = self.model.forward_decoder_detached(class_scores)
                for class_idx in range(self.args.num_classes):
                    heatmaps = heatmaps_detached.clone() # N x classes x H x W
                    single_heatmap = self.model.forward_individual_class(class_scores,class_idx) # compute individual heatmap with gradients
                    heatmaps[:,class_idx] = single_heatmap
                    loss, loss1, loss2= self.loss_function(class_scores,class_labels,heatmaps,sem_gts,sem_gt_exist)
                    if loss1!=0:
                        loss1.backward(retain_graph=True) # backpropagate heatmap loss individually
                loss2.backward() # backpropagate classification loss
            else:
                if self.args.only_send_labeled_data: #only design for ablation test
                    data=data[sem_gt_exist==True]
                    class_labels=class_labels[sem_gt_exist==True]
                    sem_gts=sem_gts[sem_gt_exist==True]
                class_scores,heatmaps = self.model(data,class_labels)
                loss, loss1, loss2= self.loss_function(class_scores,class_labels,heatmaps,sem_gts,sem_gt_exist)
                
                loss.backward()
            
            train_loss += loss.item()
            with torch.no_grad():
                if self.args.wandb != 'None':
                    watch.update(self.model,iter=self.seen_train_imgs)

            self.optimizer.step()
    
                   
            #######################
#             interest_weight1=self.model.encoder.backbone.features[0].weight.grad
#             interest_weight2=self.model.encoder.backbone.features[2].weight.grad
#             interest_weight3=self.model.encoder.backbone.classifier[6].weight.grad
#             if interest_weight1 is not None and not torch.all(interest_weight1==0):
#                 print("1111111"*7)
#                 print(interest_weight1.data.max(),"feature[0].grad for conv output")
#             if interest_weight2 is not None and not torch.all(interest_weight2==0):
#                 print("2222222"*7)
#                 print(interest_weight2.data.max(),"feature[2].grad for conv output")
#             if interest_weight3 is not None and not torch.all(interest_weight3==0):
#                 print("3333333"*7)
#                 print(interest_weight3.data.max(),"classifier[6].grad for conv output")
            #########################
            # print(f"The range of heatmap value {heatmaps[0].max()}, {heatmaps[0].min()}")
            # log_heatmaps(heatmaps[0].detach().cpu(),self.data.training_dataset.new_classes,200,title="heatmap")
            ####################

            # 3) detach all computational graphs for further postprocessing
            with torch.no_grad():
                
                # 4) get masks scores based on predicted classes
                mask_scores,_,adjusted_heatmap = self.inferring_mask_scores(class_scores,heatmaps)

                # 5) compute and log metrics 
                metrics=update_metrics(metrics,class_scores,class_labels,adjusted_heatmap,mask_scores,sem_gts)

                # 6) check the confidence map
                if self.args.wandb != 'None':
                    wandb_utils.log_training(self.seen_train_imgs,epoch,batch_idx,num_samples,
                        loss2.item(),loss1.item(),metrics,
                        data,sem_gts,heatmaps,mask_scores,self.data.training_dataset.new_classes,
                        class_labels,class_scores,self.args.log_imgs,media_log_interval=3000)
        
                self.seen_train_imgs += num_samples
                
                print(f"epoch {epoch}, self.seen_train_imgs {self.seen_train_imgs}")
                remain=self.seen_train_imgs //3000
                if self.seen_train_imgs<remain*3000+self.args.batch_size:
                    self.test(epoch)


        train_loss=train_loss/ len(self.train_loader.dataset)
        gen_log_message("train",epoch,train_loss , metrics,len(self.train_loader.dataset),print_class_iou=False)



        
    def test(self, epoch):
        
        self.model.eval()
        # init parameters etc. for tracking
        test_loss = 0
   
        metrics=init_metrics(self.test_metrics,self.args.num_classes)

        storing_examples = wandb_utils.storing_imgs(n_batches = len(self.data.testing_dataset),
                                                    test_batch_size=self.args.batch_size,seed=0)

        with torch.no_grad():
            for batch_idx, (data, class_labels,sem_gts,sem_gt_exists) in enumerate(self.test_loader):

                data, class_labels, sem_gts = data.to(self.device),class_labels.to(self.device), sem_gts.long().to(self.device)

              
                class_scores,heatmaps = self.model(data,class_labels)
                tloss, test_loss1, test_loss2 = self.loss_function(class_scores, class_labels,heatmaps,sem_gts,sem_gt_exists)
                test_loss += tloss.item()

                # get masks scores based on predicted classes
                mask_scores,_,adjusted_heatmap = self.inferring_mask_scores(class_scores,heatmaps)
            
                # calculate the metrics and update# compute metrics 
                metrics=update_metrics(metrics,class_scores,class_labels,adjusted_heatmap,mask_scores,sem_gts)

                # save images and heatmaps for logging
                if self.args.wandb != 'None':
                    storing_examples.update(batch_idx, data.detach().cpu(),sem_gts.detach().cpu(),mask_scores)    


        test_loss /= len(self.test_loader.dataset)
        old_mIoU,old_iou_class,mIoU,iou_class,match_ratio,avg_class_acc,class_acc,pixel_acc,ap = gen_log_message("test",epoch,test_loss, metrics,len(self.test_loader.dataset),return_value=True,print_class_iou=True)


    
        if self.args.wandb != 'None':
            wandb_utils.log_testing(self.seen_train_imgs,
                                    test_loss,old_mIoU,mIoU,match_ratio,pixel_acc,ap,
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
        # if pretrain_weight_name[-2:]=='21':
        #     print("WARNING: using old pretrain weight")
        #     checkpoint=self.align_checkpoint_dict(pretrain_weight_name)
           
        # else:
        #     checkpoint =torch.load(pretrain_weight_name + '.pth')
    
        # # print(checkpoint.keys())
        # # print("======================="*3)
        # # print(self.model.state_dict().keys())
     
        # if self.args.ablation_test:
        #     # if do ablation test, only load the encoder pretrain_weights
        #     encoder_checkpoint=self.delete_decoder_weights_checkpoint(checkpoint)
        #     self.model.encoder.load_state_dict(encoder_checkpoint)     
        # else:
        #     self.model.load_state_dict(checkpoint)
        
        self.load_pretrain_model_variant(pretrain_weight_name)

        return 
        
 





