from __future__ import print_function
from argparse import ArgumentDefaultsHelpFormatter
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
import sys
from torch.utils.tensorboard import SummaryWriter
sys.path.append('../')
from baseline_arch.arch_unet_vgg import create_equivalent_unet_model

from matplotlib import pyplot as plt 

from datasets import MNIST, EMNIST, FashionMNIST, Lizard, PASCAL
from datasets import PASCAL_dataset
import numpy as  np
from torch.autograd import Variable
import os
from utils.plot_utils import *
from utils.metrics_utils import *
from utils.loss_utils import *
from utils.sampling_utils import *
import torch.utils.data.sampler as sampler

# set up wandb 
import utils.wandb_utils as wandb_utils
from utils.wandb_utils import log_heatmaps
from collections import OrderedDict

# --------------------------------------
# Main function for training and test
# --------------------------------------
class Baseline(object):
    '''
    This is implementation for multi-task UNet and UNet' training and testing.

    '''
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if args.cuda else "cpu")
        self._init_dataset()

        # setup universal counter for seen training imgs
        self.seen_train_imgs = 0
        self.max_mIoU = 0 # for model saving
        self.max_mIoU_iter= 0 # for model saving

        # prepare the dataset for (pre-)train and (pre-)test
        # NOTE: when we initialize the trainloader in that way, we always have 
        # the same seed randomized order of images thus same training batches  
        # self.train_loader = self.data.train_loader 
        # self.test_loader = self.data.test_loader
        if args.add_classification==False:
            #full supervision mode
            print(f"The model is trained in FULL supervised manner with {len(self.data.training_dataset.idx_mask)} images")
            args.batch_size=min(len(self.data.training_dataset.idx_mask),args.batch_size)
            # set do evaluation after every 15 epochs for 20 n-masks
            # num_samples_list={'20':20*15,"100":100*5,"500":500*2}
            # if args.n_masks in [20, 100, 500]:
            #     self.train_loader=torch.utils.data.DataLoader(self.data.training_dataset,batch_size=args.batch_size,
            #                                 sampler=sampler.RandomSampler(data_source=self.data,
            #                                 replacement=True,
            #                                 num_samples=num_samples_list[str(args.n_masks)]),
            #                                 shuffle=False)
            # else:
            self.train_loader=torch.utils.data.DataLoader(self.data.training_dataset,batch_size=args.batch_size,
                                            shuffle=True)
 
        else:
            #semi supervision mode
            print(f"The model is trained in SEMI supervised manner with {len(self.data.training_dataset.idx_mask)} images")
            self.train_sampler = semi_supervised_sampler(self.data.training_dataset.idx_mask,
                                                        self.data.training_dataset.idx_no_mask,
                                                        args.batch_size,
                                                        len(self.data.training_dataset),
                                                        args.prob_sample_mask,
                                                        args.fluct_masks_batch,
                                                        args.seed)
            self.train_loader=torch.utils.data.DataLoader(self.data.training_dataset,sampler=self.train_sampler.sample(),batch_size=args.batch_size)
       
        
        self.test_loader=torch.utils.data.DataLoader(self.data.testing_dataset, batch_size=40)
        self._init_params(args)
        # model
        if args.model=="unet":
          
            print(f"Import Unet model from smp library and use the backbone {args.backbone} with the default pretrain weight from imagenet")
            self.args.add_classification=False
            print(f"The add_classifiation is automatically set to False as you choose model unet")
            import segmentation_models_pytorch as smp
            self.model=smp.Unet(
                encoder_name=args.backbone,        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
                encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
                in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
                classes=args.num_classes, 
                )
            # else:
            #     self.model=create_equivalent_unet_model(self.img_channel, args.backbone, self.input_size, args.num_classes)

        elif args.model== "multi_task_unet":
            from baseline_arch.multi_task_unet import MTUNet
            # backbone_name={'vgg1':'vgg16','vggbn1':'vgg16_bn', 
            #     'resnet18':'resnet18','resnet34':'resnet34',
            #     'resnet50':'resnet50','resnet101':'resnet101'}
            # encoder_n=backbone_name[args.backbone]
            self.model=MTUNet (self.img_channel, args.backbone,  args.num_classes, args.add_classification,
                args.no_skip_connection, args.fully_symmetric_unet, args.concate_before_block)
        else:
            raise NotImplementedError

        self.model.to(self.device)
        #self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=args.lr)

        # loss desgin
        self.alpha = args.loss_impact_seg #hyperparam for heatmap loss
        self.beta = args.loss_impact_bottleneck #hyperparam for classification loss
        self.seg_loss_mode = args.seg_loss_mode # 'all' # 'BCE' # 'min' # 'only_bg'
        self.file_name=f"{self.seg_loss_mode}_lr_{args.lr}" 

        # heatmap save folder
        self.heatmap_save_folder=f"results/heatmaps/{self.args.dataset}_{self.args.num_classes}/{self.file_name}/{args.model}_{args.backbone}_{args.n_masks}_{args.normalize}/"
        os.makedirs(self.heatmap_save_folder,exist_ok=True)
        
        # tensorboard writers
        # os.makedirs(f"runs/{self.args.dataset}",exist_ok=True)
        # run_name1=f"runs/{self.args.dataset}_{self.args.num_classes}/{self.file_name}/{args.model}_{args.backbone}_{args.n_masks}_{args.normalize}_train"
        # run_name2=f"runs/{self.args.dataset}_{self.args.num_classes}/{self.file_name}/{args.model}_{args.backbone}_{args.n_masks}_{args.normalize}_val"
        # self.tb_writer1=SummaryWriter(run_name1)# for train_data
        # self.tb_writer2=SummaryWriter(run_name2)# for val_data

        # get the weight from seg_gt
        w_seg_gt=self.data.training_dataset.weight_from_seg_gt()
        w_seg_gt=1/w_seg_gt
        w_seg_gt[0]=0
        self.w_seg_gt=w_seg_gt/w_seg_gt.sum()

        # check prediction
        if args.save_confidence_map:
            self.confidence_map_save_folder=f"results/confidence_map/{self.args.dataset}_{self.args.num_classes}/{self.file_name}/{args.model}_{args.backbone}_{args.normalize}/"
            os.makedirs(self.confidence_map_save_folder,exist_ok=True)

            
        # metrics
        # self.train_metrics={"classification": ["match_ratio","class_acc"],
        #                 "segmentation":["old_iou"]
        #     }
        # self.test_metrics={"classification": ["match_ratio","class_acc"],
        #                 "segmentation":["old_iou", "iou","AP", "pixel_acc"]
        #     }

        self.train_metrics={"classification": ["match_ratio","class_acc"],
                        "segmentation":["iou"]
            }
        self.test_metrics={"classification": ["match_ratio","class_acc"],
                        "segmentation":["iou"]
            }
  
        if args.use_earlystopping:
            self.early_stopping=EarlyStopping(patience=4)
            print("Applying early stopping with patience-epochs=4")


    def _init_params(self,args):
        if args.dataset in ['MNIST','EMNIST','FashionMNIST']:
            self.img_channel=1
            self.input_size=(1,28,28)
        elif args.dataset in ['Lizard','PASCAL']:
            self.img_channel=3
            self.input_size=(3,256,256)
        else:
            print("Dataset not supported")

    def _init_dataset(self):
        if self.args.dataset == 'MNIST':
            self.data = MNIST(self.args)
        elif self.args.dataset == 'EMNIST':
            self.data = EMNIST(self.args)
        elif self.args.dataset == 'FashionMNIST':
            self.data = FashionMNIST(self.args)
        elif self.args.dataset== 'Lizard':
            self.data=Lizard(self.args)
        elif self.args.dataset == 'PASCAL':
            self.data = PASCAL(self.args)
        else:
            print("Dataset not supported")

    def align_checkpoint_dict(self, pretrain_weight_name,domain_a, domain_b):
        checkpoint =torch.load(pretrain_weight_name + '.pth')
        # standardize the checkpoint keys
        cl_head_checkpoint=OrderedDict()      
        # filter out the decoder layers checkpoint
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
        

        # if domain_a=="smp_unet" and domain_b=="unrolled_xai_checkpoint":
        #     pass

        # # Deprecated  part 2023.08.26
        # elif domain_a=="self_define_unet" and domain_b=="unrolled_xai_checkpoint":
           
        #     mapping=self.get_mapping_list(self.args.backbone)
      
        #     for m in mapping:
        #         checkpoint[m[0]+".weight"]=checkpoint[m[1]+".weight"]
        #         checkpoint[m[0]+".bias"]=checkpoint[m[1]+".bias"]
        #         del checkpoint[m[1]+".weight"],checkpoint[m[1]+".bias"]

        #         if m[1]+".running_mean" in checkpoint.keys():
        #             checkpoint[m[0]+".running_mean"]=checkpoint[m[1]+".running_mean"]
        #             checkpoint[m[0]+".running_var"]=checkpoint[m[1]+".running_var"]
        #             checkpoint[m[0]+".num_batches_tracked"]=checkpoint[m[1]+".num_batches_tracked"]
        #             del checkpoint[m[1]+".running_mean"],checkpoint[m[1]+".running_var"],checkpoint[m[1]+".num_batches_tracked"]

                
        # else:
        #     raise NotImplementedError
        # print("encoder checkpoint keys:", checkpoint.keys())
        # print("classification head checkpoint keys:" ,cl_head_checkpoint.keys())
        return checkpoint,cl_head_checkpoint

    def load_pretrain_model(self,pretrain_weight_name):
        if self.args.pretrain_weight_name !="imagenet":
            cp_epoch=int(pretrain_weight_name.split("/")[-1].split("_")[1])
            print(f"Loard pretrained weights {pretrain_weight_name + '.pth'} at check point epoch={cp_epoch}")
        
        # check if you are using the right pretrain_weight and right backbone
        mapping_name={'vgg':['vgg16'],'vggbn':['vgg16_bn'], 
            'resnet18':['resnet18'],'resnet34':['resnet34'],
            'resnet50':['resnet50'],'resnet101':['resnet101']}
        pretrain_weight_backbone_name=pretrain_weight_name.split("/")[-1].split("_")[0]
        if self.args.backbone not in mapping_name[pretrain_weight_backbone_name]:
            raise ValueError(f'backbone name {self.args.backbone} and pretrain backbone name {pretrain_weight_backbone_name} is not matching.')

        
        if self.args.pretrain_weight_name !="imagenet":
            checkpoint,cl_head_checkpoint=self.align_checkpoint_dict(pretrain_weight_name,domain_a="smp_unet",domain_b="unrolled_xai_checkpoint")
            self.model.encoder.load_state_dict(checkpoint)
            if self.model.classification_head is not None:
                self.model.classification_head.load_state_dict(cl_head_checkpoint)


        else:
            pass
            # else:
            #     if self.args.pretrain_weight_name !="imagenet":
            #         checkpoint=self.align_checkpoint_dict(pretrain_weight_name,domain_a="self_define_unet",domain_b="unrolled_xai_checkpoint")
            #         self.model.load_state_dict(checkpoint,strict=False)
            #         print("Warning: You are using load self.model.load_state_dict(checkpoint,strict=False)")
            #     else:
            #         raise NotImplementedError

        # self.seen_train_imgs = cp_epoch * len(self.train_loader.dataset) 
        self.seen_train_imgs = cp_epoch * 15676 # The number of images with class level labels= 15676

    def save_model(self,epoch,keyword):
        torch.save(self.model.state_dict(),f"snapshot/{self.args.model}_{epoch}_{keyword}_{self.args.num_classes}.pth")


    def generate_class_weights(self, sem_gts):
        weights = torch.stack([ torch.sum(sem_gts==i,axis=(1,2)) for i in range(self.args.num_classes) ],dim=1) 
        weights=1/torch.sum(weights,dim=0)
        weights[weights == float('inf')] = 0
        return weights


    # def add_bg_heatmap(self,heatmaps):
    #     if self.args.num_classes==6:
    #         N,_,H,W=heatmaps.shape
    #         bg=torch.zeros((N,1,H,W)).to(self.device)
    #         heatmaps=torch.cat((bg, heatmaps),dim=1) #(N, 7, H, W)  

    #     return heatmaps    


    def loss_function(self, pred_class_scores, class_gt, heatmaps,seg_gt,seg_gt_exists):

        hm_sf=self.args.heatmap_scale # 1000
        hm_offset=self.args.heatmap_offset # 0 #-1e-05
        bg_weight_BCE_hm_loss=self.args.bg_weight_in_BCE_hm_loss
        nonexist_weight_BCE_hm_loss=self.args.nonexist_weight_in_BCE_hm_loss

        # heatmap loss
        heatmap_loss=0 
        if self.seg_loss_mode=="min":
            weights = (1-torch.concat((torch.ones((class_gt.size(0),1)).to(self.device),class_gt),dim=1))
            heatmap_loss=nn.CrossEntropyLoss(weight=weights,reduction="sum")(input=hm_sf*heatmaps,target=0*seg_gt)
        elif self.seg_loss_mode=="BCE":
            ### try heatmap-wise BCE:
            seg_gt_one_hot=F.one_hot(torch.clamp(seg_gt,min=0),num_classes=self.args.num_classes).permute(0,3,1,2).float().to(self.device)
            weight_by_gt_exists=torch.zeros(heatmaps.size()).to(self.device)
            for batch_idx in range(heatmaps.size(0)):
                if seg_gt_exists[batch_idx]:
                    weight_by_gt_exists[batch_idx,0]=bg_weight_BCE_hm_loss
                    for class_idx in range(1,self.args.num_classes):
                        if class_gt[batch_idx,class_idx]==0:
                            weight_by_gt_exists[batch_idx,class_idx]=nonexist_weight_BCE_hm_loss/self.args.num_classes
                        else:
                            weight_by_gt_exists[batch_idx,class_idx]=1.0
            img_size=heatmaps.size(2)*heatmaps.size(3)
            if self.args.use_w_seg_gt:
                weight_by_gt_exists=weight_by_gt_exists*self.w_seg_gt[None,:,None,None].to(self.device)
            sum_hm_weights=torch.clamp(torch.sum(weight_by_gt_exists),min=1.0)
            # print("sum_hm_weights:")
            # print(sum_hm_weights,flush=True)
            heatmap_loss=nn.BCEWithLogitsLoss(weight=weight_by_gt_exists,reduction='sum')(input=hm_sf*(heatmaps+hm_offset),target=seg_gt_one_hot)/sum_hm_weights
            heatmap_loss*=self.args.batch_size
        elif self.seg_loss_mode in ["all", "softmax"]:
            weights=self.generate_class_weights(seg_gt)
            heatmap_loss=nn.CrossEntropyLoss(reduction="mean",ignore_index=-1)(input=hm_sf*(heatmaps+hm_offset),target=seg_gt) # TODO check if seg_gt right format, introduce back weights
      

        else:
            raise NotImplementedError("This segmentation loss mode has not been implemented.")

        if self.args.add_classification:
            weight = torch.ones(self.args.num_classes).to(self.device)
            weight[0] = 0
            classification_loss= F.multilabel_soft_margin_loss(pred_class_scores,class_gt.float(),weight=weight) # size (N, num_classes) flat tensor
            classification_loss*=self.args.batch_size

            # total loss
            t_loss = self.beta*classification_loss+self.alpha*heatmap_loss
            return t_loss, self.alpha*heatmap_loss, self.beta*classification_loss
        else:
            t_loss = heatmap_loss

            return t_loss, t_loss, torch.tensor(0)
       

    def inferring_mask_scores(self,class_pred,heatmap_pred,class_thresh=0.5,method='agf'):
        '''
        inferring mask scores (NxCxHxW) with values 0,..,1
        from heatmaps (NxCxHxW) [-infty,infty]
        based on the prediction of class
        '''
        class_pred=class_pred.detach().cpu()
        heatmap_pred=heatmap_pred.detach().cpu()

        # get predicted classes
        binary_pred = torch.sigmoid(class_pred).gt(class_thresh).long()

        # infer mask
        adjusted_heatmap = heatmap_pred.clone()

        if method == 'agf':  
            adjusted_heatmap = adjusted_heatmap.clamp(min=0)                                                  # exclude negative relevance
            adjusted_heatmap[:, 1:] += -0.1 * (1-binary_pred[:, 1:, None, None].float())        # if predicted downscaled (why only 0.1??)
            mask_pred = F.softmax(adjusted_heatmap, 1)
        elif method == 'sigmoid_per_predicted_class':
            adjusted_heatmap[binary_pred==0]= self.args.heatmap_offset-1
            adjusted_heatmap[:,0] = self.args.heatmap_offset
            mask_pred=torch.sigmoid(adjusted_heatmap).gt(class_thresh).long()
            # mask_pred=torch.argmax(hm_sigmoid,dim=1).long() # todo: this crashes
            # print("mask_pred size sigmoid_per_predicted_class:")
            # print(mask_pred.size(),flush=True)
        elif method == 'bg_0':
            adjusted_heatmap[binary_pred==0] = self.args.heatmap_offset-1
            adjusted_heatmap[:,0] = self.args.heatmap_offset
            mask_pred = F.softmax(adjusted_heatmap, 1)
            # print("mask_pred size bg_0:")
            # print(mask_pred.size(),flush=True)
        else:
            mask_pred = F.softmax(adjusted_heatmap, 1)
        
        return mask_pred,binary_pred,adjusted_heatmap



    # --------------------------------------
    # Functions for offical train
    # --------------------------------------
        
    def train(self, epoch):
        self.model.train()
        train_loss = 0
        # class_counts = torch.zeros(self.args.num_classes).to(self.device) # TODO put into dedicated function - push to init
        metrics=init_metrics(self.train_metrics,self.args.num_classes)
        watch = wandb_utils.watch_gradients()
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
            # print(f"Train batch idx {batch_idx}: heatmap-loss: { loss1.data:.4f} classification-loss: { loss2.data}")
            
            train_loss += loss.item()
            with torch.no_grad():
                if self.args.wandb != 'None':
                    watch.update(self.model,iter=self.seen_train_imgs)
            self.optimizer.step()
            
  
            # print(f"Train batch idx {batch_idx}: loss: {loss.data}")
            # 3) detach all computational graphs for further postprocessing
            with torch.no_grad():
                
                # 4) get masks scores based on predicted classes
                mask_scores,_,adjusted_heatmap = self.inferring_mask_scores(class_scores,heatmaps,method=self.args.mask_inference_method)
                # calculate the metrics and update
                metrics=update_metrics(metrics,class_scores,class_labels,adjusted_heatmap,mask_scores,sem_gts)
            
                if self.args.wandb != 'None':
                    wandb_utils.log_training(self.seen_train_imgs,epoch,batch_idx,len(data),
                        loss2.item(),loss1.item(),metrics,
                        data,sem_gts,heatmaps,mask_scores,self.data.training_dataset.new_classes,
                        class_labels,class_scores,self.args.log_imgs,media_log_interval=3000)  #todo metrics is not used in wandb_utils.log_training
         
                self.seen_train_imgs += len(data)
                #save heatmaps
                # if batch_idx==0:
                #     save_raw_heatmap_sem_gt(4,2,data,heatmaps[:8],sem_gts[:8], f"{epoch}_train_heatmap.png" , self.heatmap_save_folder,normalize=self.args.normalize)

                if self.args.add_classification==False:
                    remain=self.seen_train_imgs //1500
                    if self.seen_train_imgs<remain*1500+self.args.batch_size:
                        self.test(epoch)
                else:
                    remain=self.seen_train_imgs //3000
                    if self.seen_train_imgs<remain*3000+self.args.batch_size:
                        self.test(epoch)


                if self.args.save_confidence_map and batch_idx==0:
                    save_raw_confidencemap_sem_gt(4, 2,data[:8],heatmaps[:8],sem_gts[:8], f"{epoch}_{batch_idx}_train_confidence_map.png", 
                                    self.confidence_map_save_folder, color=False, normalize=self.args.normalize)


        train_loss=train_loss/ len(self.train_loader.dataset)
        gen_log_message("train",epoch,train_loss , metrics,len(self.train_loader.dataset),print_class_iou=False)
        # write_summaries(self.tb_writer1, train_loss , metrics,len(self.train_loader.dataset), epoch+1)

        
    def test(self, epoch):
        
        self.model.eval()
        metrics=init_metrics(self.test_metrics,self.args.num_classes)
        # init parameters etc. for tracking
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


                # print(f"Test batch idx {batch_idx}: loss: { tloss.data: .4f}")
                # get masks scores based on predicted classes
                mask_scores,_,adjusted_heatmap = self.inferring_mask_scores(class_scores,heatmaps,method=self.args.mask_inference_method)


                # calculate the metrics and update# compute metrics 
                metrics=update_metrics(metrics,class_scores,class_labels,adjusted_heatmap,mask_scores,sem_gts)
            
                # save images and heatmaps for logging
                if self.args.wandb != 'None':
                    storing_examples.update(batch_idx, data.detach().cpu(),sem_gts.detach().cpu(),mask_scores)  

                # save confidence maps
                if self.args.save_confidence_map and batch_idx==0:
                    save_raw_confidencemap_sem_gt(4, 2,data[:8],heatmaps[:8],sem_gts[:8], f"{epoch}_{batch_idx}_test_confidence_map.png", 
                                    self.confidence_map_save_folder, color=False, normalize=self.args.normalize)

                #save heatmaps
                # if batch_idx==0:
                #     save_raw_heatmap_sem_gt(4,2,data,heatmaps[:8],sem_gts[:8], f"{epoch}_test_heatmap.png" , self.heatmap_save_folder,normalize=self.args.normalize)
        
        test_loss /= len(self.test_loader.dataset)
        old_mIoU,_,mIoU,_,match_ratio,avg_class_acc,_,pixel_acc,ap = gen_log_message("test",epoch,test_loss, metrics,len(self.test_loader.dataset),return_value=True,print_class_iou=True)
        # write_summaries(self.tb_writer2,test_loss,metrics,len(self.test_loader.dataset),epoch)
        
        
        if self.args.wandb != 'None':
            wandb_utils.log_testing(self.seen_train_imgs,
                                    test_loss,old_mIoU,mIoU,match_ratio,pixel_acc,ap,
                                    storing_examples.get_imgs(),storing_examples.get_gt_masks(),storing_examples.get_pred_masks(),self.data.training_dataset.new_classes)  

        if self.args.save_all_checkpoints:
            if self.args.pretrain_weight_name is not None:
                pretrained_model_name = self.args.pretrain_weight_name.split("/")[-1]
                pre_epochs=int(pretrained_model_name.split("_")[1])
            else:
                pre_epochs=0

            # set name
            if self.args.model== "multi_task_unet" and self.args.add_classification==True:
                name="mt_unet"
            elif self.args.model=="multi_task_unet" and self.args.add_classification==False:
                name="std_unet"
            else:
                name=self.args.model


            # save the model
            os.makedirs(f"{self.args.save_folder}"+"all_checkpoints/"+f"{self.args.backbone}_seed{self.args.seed}_pre_epochs{pre_epochs}",exist_ok=True)
            torch.save(self.model.state_dict(), 
                f"{self.args.save_folder}"+"all_checkpoints/"+f"{self.args.backbone}_seed{self.args.seed}_pre_epochs{pre_epochs}"+
                    f"/{name}_{self.args.backbone}_iter{self.seen_train_imgs}_lab{self.args.n_masks}_lr{self.args.lr}_bs{self.args.batch_size}.pth")

        else:
            if mIoU >= self.max_mIoU and self.args.save_folder is not None:
                # record the pret training epochs
                if self.args.pretrain_weight_name is not None:
                    pretrained_model_name = self.args.pretrain_weight_name.split("/")[-1]
                    pre_epochs=int(pretrained_model_name.split("_")[1])
                else:
                    pre_epochs=0

                # set name
                if self.args.model== "multi_task_unet" and self.args.add_classification==True:
                    name="mt_unet"
                elif self.args.model=="multi_task_unet" and self.args.add_classification==False:
                    name="std_unet"
                else:
                    name=self.args.model
                # save the model
                os.makedirs(f"{self.args.save_folder}/{self.args.backbone}_seed{self.args.seed}_pre_epochs{pre_epochs}",exist_ok=True)
                torch.save(self.model.state_dict(), 
                    f"{self.args.save_folder}"+f"{self.args.backbone}_seed{self.args.seed}_pre_epochs{pre_epochs}"+
                        f"/{name}_{self.args.backbone}_iter{self.seen_train_imgs}_lab{self.args.n_masks}_lr{self.args.lr}_bs{self.args.batch_size}.pth")
                # update mIoU
                try:
                    os.remove(f"{self.args.save_folder}"+f"{self.args.backbone}_seed{self.args.seed}_pre_epochs{pre_epochs}"+
                            f"/{name}_{self.args.backbone}_iter{self.max_mIoU_iter}_lab{self.args.n_masks}_lr{self.args.lr}_bs{self.args.batch_size}.pth")
                except FileNotFoundError:
                    pass
                self.max_mIoU= mIoU
                self.max_mIoU_iter=self.seen_train_imgs

        
        self.model.train()

        
 
    # Deprecated  part 2023.08.26
    def get_mapping_list(self,backbone):
        if backbone=="vgg1":
            mapping=[
            ["0.l_conv.0.conv_pass.0","features.0"],["0.l_conv.0.conv_pass.2", "features.2"],
            ["0.l_conv.1.conv_pass.0","features.5"],["0.l_conv.1.conv_pass.2", "features.7"],
            ["0.l_conv.2.conv_pass.0","features.10"],["0.l_conv.2.conv_pass.2", "features.12"],["0.l_conv.2.conv_pass.4", "features.14"],
            ["0.l_conv.3.conv_pass.0","features.17"],["0.l_conv.3.conv_pass.2", "features.19"],["0.l_conv.3.conv_pass.4", "features.21"],
            ["0.l_conv.4.conv_pass.0","features.24"],["0.l_conv.4.conv_pass.2", "features.26"],["0.l_conv.4.conv_pass.4", "features.28"]
            ]
            return mapping
        elif backbone=="vggbn1":
            mapping=[
            ["0.l_conv.0.conv_pass.0","features.0"],["0.l_conv.0.conv_pass.1", "features.1"],
            ["0.l_conv.0.conv_pass.3","features.3"],["0.l_conv.0.conv_pass.4", "features.4"],# layer1
            ["0.l_conv.1.conv_pass.0","features.7"],["0.l_conv.1.conv_pass.1", "features.8"],
            ["0.l_conv.1.conv_pass.3","features.10"],["0.l_conv.1.conv_pass.4", "features.11"],#layer2
            ["0.l_conv.2.conv_pass.0","features.14"],["0.l_conv.2.conv_pass.1", "features.15"],
            ["0.l_conv.2.conv_pass.3","features.17"],["0.l_conv.2.conv_pass.4", "features.18"],
            ["0.l_conv.2.conv_pass.6","features.20"],["0.l_conv.2.conv_pass.7", "features.21"],#layer3
            ["0.l_conv.3.conv_pass.0","features.24"],["0.l_conv.3.conv_pass.1", "features.25"],
            ["0.l_conv.3.conv_pass.3","features.27"],["0.l_conv.3.conv_pass.4", "features.28"],
            ["0.l_conv.3.conv_pass.6","features.30"],["0.l_conv.3.conv_pass.7", "features.31"],#layer4
            ["0.l_conv.4.conv_pass.0","features.34"],["0.l_conv.4.conv_pass.1", "features.35"],
            ["0.l_conv.4.conv_pass.3","features.37"],["0.l_conv.4.conv_pass.4", "features.38"],
            ["0.l_conv.4.conv_pass.6","features.40"],["0.l_conv.4.conv_pass.7", "features.41"],#layer5
            ]
            return mapping
        else:
            raise NotImplementedError


