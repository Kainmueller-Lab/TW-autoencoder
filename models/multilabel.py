from __future__ import print_function
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
import sys
from torch.utils.tensorboard import SummaryWriter
sys.path.append('../')


# from arch.arch_MNIST import  CNN_Decoder,CNN_Encoder # encoder and decoder for MNIST dataset
# from arch.arch_vgg import CNN_Encoder, CNN_Decoder
from arch.architectures_utils import FwdHooks
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
import wandb


# set up wandb 
import utils.wandb_utils as wandb_utils
from utils.wandb_utils import log_heatmaps

# TODO at network for only classification 
# TODO make option of loading weights into pretrained XAI model

# --------------------------------------
# Network for multi-label classification
# --------------------------------------
class Network(nn.Module):
    def __init__(self, args):
        super(Network, self).__init__()
        
        self.num_classes=args.num_classes
        self._init_params(args)
        self.use_old_xai=args.use_old_xai
    
        CNN_Encoder,CNN_Decoder=self.import_arch(args.backbone)
        self.encoder = CNN_Encoder(args.backbone, self.img_channel, self.input_size, args.num_classes)

        #initialize the hooks pocket and register hooks for the layers inside the encoder
        self.fwd_hooks=FwdHooks(args.xai)
        self.register_hooks(self.encoder)
        
        # run test_forward for assigning the attribute of in_tensor.shape 
        # for the transposeconv_2d layer in the decoder
        self.test_forward(self.encoder, self.input_size)
        self.tied_weights_decoder = CNN_Decoder(self.encoder, args.xai, args.epsilon, args.alpha, memory_efficient=args.memory_efficient,
            detach_bias=args.detach_bias,
            normal_relu=args.normal_relu, normal_deconv=args.normal_deconv,
            normal_unpool=args.normal_unpool,
            multiply_input=args.multiply_input,remove_heaviside=args.remove_heaviside,
            remove_last_relu=args.remove_last_relu, add_bottle_conv=args.add_bottle_conv)

        self.normal_deconv=args.normal_deconv

        if args.memory_efficient:
            print(f"Attention: This autoencoder uses memory efficient mode")

        

    def import_arch(self, keyword):
        if self.use_old_xai:
            
            if keyword=="CNN_MNIST":
                from old_arch.arch_MNIST import  CNN_Decoder,CNN_Encoder # encoder and decoder for MNIST dataset
            elif keyword=="vgg":
                print("Using old xai to do sanity check")
                from old_arch.arch_vgg import  CNN_Decoder,CNN_Encoder # encoder and decoder for MNIST dataset
            elif keyword=="efficientnet":
                from old_arch.arch_efficientnet import  CNN_Decoder,CNN_Encoder # encoder and decoder for MNIST dataset
            else:
                raise NotImplemented
            return CNN_Encoder,CNN_Decoder

        else:
            if keyword=="CNN_MNIST":
                from arch.arch_MNIST import  CNN_Decoder,CNN_Encoder # encoder and decoder for MNIST dataset
            elif keyword=="vgg16":
                from arch.arch_vgg import  CNN_Decoder,CNN_Encoder # encoder and decoder for MNIST dataset
            elif keyword=="vgg16_bn":
                from arch.arch_vggbn import  CNN_Decoder,CNN_Encoder # encoder and decoder for MNIST dataset
            elif keyword=="efficientnet":
                from arch.arch_efficientnet import  CNN_Decoder,CNN_Encoder # encoder and decoder for MNIST dataset
            elif keyword in ["resnet50" ,"resnet101"] :
                # from arch.arch_resnet50 import  CNN_Decoder,CNN_Encoder # encoder and decoder for MNIST dataset
                from arch.resnet50_new import  CNN_Decoder,CNN_Encoder # encoder and decoder for MNIST dataset
            elif keyword in ["resnet18" ,"resnet34"] :
                from arch.arch_resnet18 import  CNN_Decoder,CNN_Encoder # encoder and decoder for MNIST dataset
            else:
                raise NotImplemented
            return CNN_Encoder,CNN_Decoder


    def _init_params(self,args):
        if args.dataset in ['MNIST','EMNIST','FashionMNIST']:
            self.img_channel=1
            self.input_size=(1,28,28)
        elif args.dataset in ['Lizard','PASCAL']:
            self.img_channel=3
            self.input_size=(3,256,256)
        else:
            print("Dataset not supported")
        

    def forward(self, x, targets=None, only_classification=False):
        '''
        targets: torch.tensor (N,)
        '''
        if only_classification:
            z = self.encoder(x) 
            return z
        else:
            if self.normal_deconv:
                z = self.encoder(x) 
                # random put a class_idx
                heatmaps=self.tied_weights_decoder(z,0)

            else:

                z = self.encoder(x) 
                # calculate the heatmap for all classes
                heatmaps_poket=[]
            #######################   
    #             for i in range(0,10):
    #                 print(f"decoder run for {i} class")
    #                 tmp_heatmaps=self.tied_weights_decoder(self.indi_features(z,i),i)
    #                 heatmaps_poket.append(tmp_heatmaps) # (N, H, W)
    #             for i in range(10, self.num_classes):
    #                 fake_heatmap=torch.ones((1,256,256))
    #                 fake_heatmap=fake_heatmap.to(torch.device("cuda"))
    #                 heatmaps_poket.append(fake_heatmap) # (N, H, W)
            #######################
                for i in range(self.num_classes):
                    # print(f"decoder run for {i} class")
                    tmp_heatmaps=self.tied_weights_decoder(self.indi_features(z,i),i)
                    heatmaps_poket.append(tmp_heatmaps) # (N, H, W)
                heatmaps=torch.stack(heatmaps_poket,dim=1)  # (N, num_classes, H, W)
            return z, heatmaps


    def forward_decoder_detached(self, z):  
        z = z.clone() # safety opperation
        z = z.detach() # NOTE delete not needed computational graph of z for this computation
        heatmaps_poket=[]

        for i in range(self.num_classes):
            tmp_heatmaps=self.tied_weights_decoder(self.indi_features(z,i),i)
            heatmaps_poket.append(tmp_heatmaps.detach()) # (N, H, W) NOTE here the computational graph of every tmp_heatmap gets deleted
        heatmaps=torch.stack(heatmaps_poket,dim=1)  # (N, num_classes, H, W)
        
        return heatmaps


    def forward_individual_class(self, z, class_idx):  
        heatmap=self.tied_weights_decoder(self.indi_features(z,class_idx),class_idx) # (N, H, W)  
        return heatmap


    def test_forward(self,module,input_size):
        x=Variable(torch.ones(2, *input_size)) # bacth_size must larger than 1 for batchnorm1d   
        y=module(x)     
        return


    def register_hooks(self, parent_module):
        
        for mod in parent_module.children():
            # print("LRP processing module... ", mod)     
            if list(mod.children()):
                self.register_hooks(mod)
                continue
            mod.register_forward_hook(
                self.fwd_hooks.get_layer_fwd_hook(mod))


    def indi_features(self,z,targets):
        assert z.shape[-1]==self.num_classes, "The encoder part has problem."
        if targets==None:
            tmp_list=[]         
            for i in range(self.num_classes):
                tmp_list.append(self.initialise_rel("normal",z,i))
            return torch.stack(tmp_list,dim=0)
        else:
            return self.initialise_rel("normal",z,targets)
            

    def initialise_rel(self,init_type,class_scores,targets):
        device=class_scores.device
        N=class_scores.shape[0]
        
        if init_type == "softmax":
            T=self.softmaxlayer_lrp(class_scores,targets,device)

        elif init_type == "normal":
            T = torch.zeros(class_scores.shape).to(device)
            T[range(N),targets]=class_scores[range(N),targets]


        elif init_type== "contrastive":
            T=torch.abs(class_scores)
            T[range(N),targets]=0.0

        elif init_type == "normal-contrastive":
            T = torch.ones(class_scores.shape).to(device)*(-1)
            T[range(N),targets]=1.0
            T=T*class_scores
  
        else:
            raise NotImplementedError

        return T


    def softmaxlayer_lrp(class_score,targets,device):
        '''
        Input:
            class_score: torch tensor, shape [N, num_classes], should be the logit
            targets: list of GT labels, len=N or torch.tensor with shape (N,)
            device: cpu or cuda
        Return:
            R, the relevance distribution, torch tensor, shape [N, num_classes]

        example:
            R1= Z1, target class
            R2=-Z2*exp(-(Z1-Z2))/(exp(-(Z1-Z2))+exp(-(Z1-Z3)))
            R3=-Z3*exp(-(Z1-Z3))/(exp(-(Z1-Z2))+exp(-(Z1-Z3)))
            
        principle:
            assume c is the GT label or intertesed class label
            Rc=Zc
            Rc'=-Zc'*exp(-(Zc-Zc'))/sum_{c''!=c}exp(-(Zc-Zc''))
            
        '''

        # -(Zc-Zc'')
        assert class_score.dim()==2,"The dimension of the class_score tensor shouold be 2." 
        # targets=torch.LongTensor(targets).to(device)
        targets=torch.LongTensor(targets)
        scores_diff=class_score-class_score.gather(1,targets.view(-1,1))

        # exp(Zc'-Zc)/sum_{c''!=c}exp(-(Zc-Zc''))
        R=torch.exp(scores_diff)/(torch.exp(scores_diff).sum(dim=1,keepdim=True)-1.0)

        # Rc'=-Zc'*exp(Zc'-Zc)/sum_{c''!=c}exp(-(Zc-Zc'')) &  Rc=Zc
        R=(-1.0)*class_score*R
        R[range(R.shape[0]),targets]=class_score[range(R.shape[0]),targets]

        return R

   









# --------------------------------------
# Main function for training and test
# --------------------------------------
class AE(object):
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if args.cuda else "cpu")
        self._init_dataset()

        # setup universal counter for seen training imgs
        self.seen_train_imgs = 0
        self.max_mIoU = 0 # for model saving
        self.max_mIoU_iter= 0 # for model saving

        # setup dataloaders and samplers
        self.train_sampler = semi_supervised_sampler(self.data.training_dataset.idx_mask,
                                                     self.data.training_dataset.idx_no_mask,
                                                     args.batch_size,
                                                     len(self.data.training_dataset),
                                                     args.prob_sample_mask,
                                                     args.fluct_masks_batch,
                                                     args.seed)
        self.train_loader=torch.utils.data.DataLoader(self.data.training_dataset,sampler=self.train_sampler.sample(),batch_size=args.batch_size)
        self.test_loader=torch.utils.data.DataLoader(self.data.testing_dataset, batch_size=args.batch_size)
        if args.pre_batch_size is not None and args.pre_epochs>0:
            self.pre_train_loader=torch.utils.data.DataLoader(self.data.pre_training_dataset,batch_size=args.pre_batch_size,shuffle=True)
            self.pre_test_loader=torch.utils.data.DataLoader(self.data.pre_testing_dataset, batch_size=args.pre_batch_size)

        # model
        self.model = Network(args) 
        self.model.to(self.device)
        #self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=args.lr)

        # loss desgin
        self.alpha = args.loss_impact_seg #hyperparam for heatmap loss
        self.beta = args.loss_impact_bottleneck #hyperparam for classification loss
        self.seg_loss_mode = args.seg_loss_mode # 'all' # 'BCE' # 'min' # 'only_bg'
        self.file_name=f"{self.seg_loss_mode}_alpha_{self.alpha}_beta_{self.beta}" 

        # heatmap save folder
        self.heatmap_save_folder=f"results/heatmaps/{self.args.dataset}_{self.args.num_classes}/{self.file_name}/{args.model}_{args.backbone}_{args.normalize}/"
        os.makedirs(self.heatmap_save_folder,exist_ok=True)
        
        # wandb table
        if args.log_tables:
            class_name_list=['iter']
            class_name_list.extend(list(self.data.training_dataset.new_classes.values()))
            self.class_acc_table = wandb.Table(
            columns=class_name_list)
            self.class_iou_table = wandb.Table(
            columns=class_name_list)
            # self.class_old_iou_table = wandb.Table(
            # columns=class_name_list)
            self.table_count=1


        # tensorboard writers
        # os.makedirs(f"runs/{self.args.dataset}",exist_ok=True)
        # run_name1=f"runs/{self.args.dataset}_{self.args.num_classes}/{self.file_name}/{args.model}_{args.backbone}_{args.normalize}_train"
        # run_name2=f"runs/{self.args.dataset}_{self.args.num_classes}/{self.file_name}/{args.model}_{args.backbone}_{args.normalize}_val"
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


        if args.use_earlystopping:
            self.early_stopping=EarlyStopping(patience=4)
            print("Applying early stopping with patience-epochs=4")


        # metrics
        # self.train_metrics={"classification": ["match_ratio","class_acc"],
        #                 "segmentation":["old_iou"]
            # }
        # self.test_metrics={"classification": ["match_ratio","class_acc"],
        #                 "segmentation":["old_iou","iou", "AP", "pixel_acc"]
        #     }
        self.train_metrics={"classification": ["match_ratio","class_acc"],
                        "segmentation":["iou"]
            }
        self.test_metrics={"classification": ["match_ratio","class_acc"],
                        "segmentation":["iou"]
            }

        self.pretraining_train_metrics={"classification": ["match_ratio","class_acc"]}
        self.pretraining_test_metrics={"classification": ["match_ratio","class_acc"]}

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
                checkpoint[key.replace('encoder.backbone.', 'backbone.')] = checkpoint[key]
                del checkpoint[key]

        self.model.encoder.load_state_dict(checkpoint)
   

    def load_pretrain_model(self,pretrain_weight_name,cp_epoch):
        print(f"Loard pretrained weights {pretrain_weight_name + '.pth'} at check point epoch={cp_epoch}")
        if not self.args.normal_relu and not self.args.normal_deconv and not self.args.remove_heaviside:
            # self.model.load_state_dict(torch.load(pretrain_weight_name + '.pth'), strict=True) TODO now test resnet50_new
            self.load_pretrain_model_variant(pretrain_weight_name)
        else:
            self.load_pretrain_model_variant(pretrain_weight_name)
        # self.seen_train_imgs = cp_epoch * len(self.train_loader.dataset) # TODO make general not only for plain pretraining with same dataset
        self.seen_train_imgs = cp_epoch * 15676 # The number of images with class level labels= 15676
        # save copy of all the weights with no gradients for respective weight decay
        if self.args.pretrain_weight_decay:
            self.model0 = Network(self.args)
            self.model0.load_state_dict(torch.load(pretrain_weight_name + '.pth'), strict=True)
            self.model0.to(self.device)
            self.model0.eval()


    def save_model(self,epoch,keyword):
        torch.save(self.model.state_dict(),f"snapshot/{self.args.backbone}_{epoch}_{keyword}_{self.args.num_classes}.pth")


    def generate_class_weights(self, sem_gts):
        weights = torch.stack([ torch.sum(sem_gts==i,axis=(1,2)) for i in range(self.args.num_classes) ],dim=1) 
        weights=1/torch.sum(weights,dim=0)
        weights[weights == float('inf')] = 0
        return weights


    def add_bg_heatmap(self,heatmaps):
        if self.args.num_classes==6:
            N,_,H,W=heatmaps.shape
            bg=torch.zeros((N,1,H,W)).to(self.device)
            heatmaps=torch.cat((bg, heatmaps),dim=1) #(N, 7, H, W)  

        return heatmaps    


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
        elif self.seg_loss_mode == "valley":
            seg_gt_one_hot=F.one_hot(torch.clamp(seg_gt,min=0),num_classes=self.args.num_classes).permute(0,3,1,2).float().to(self.device)         
            heatmaps = torch.abs(heatmaps)
            if torch.isnan(heatmaps).any():
                print('nan 0')
            if torch.isinf(heatmaps).any():
                print('inf 0')
            heatmaps=torch.nan_to_num(heatmaps,nan=0)
            Imask=torch.ones(seg_gt_one_hot.shape).type_as(seg_gt_one_hot)-seg_gt_one_hot

            heatmapBKG=torch.mul(heatmaps,Imask)
            heatmapBKG=self.GlobalWeightedRankPooling(heatmapBKG) # N x num_classes
            #activation:
            heatmapBKG=heatmapBKG/(heatmapBKG+1)
            #cross entropy (pixel-wise):
            heatmapBKG=torch.clamp(heatmapBKG,max=1-1e-7)
            heatmap_loss=-torch.log(torch.ones(heatmapBKG.shape).type_as(heatmapBKG)-heatmapBKG) # N x num_classes

            # only pick the heatmaps with seg_gt
            heatmap_loss= heatmap_loss[seg_gt_exists,:] # N(with seg_gt) x num_classes
            heatmap_loss=torch.sum(heatmap_loss)*heatmaps.shape[-1]*heatmaps.shape[-2] #channels and classes mean         
        elif self.seg_loss_mode in ["all", "softmax"]:
            # weights=self.generate_class_weights(seg_gt)
            heatmap_loss=nn.CrossEntropyLoss(reduction="mean",ignore_index=-1)(input=hm_sf*(heatmaps+hm_offset),target=seg_gt) # TODO check if seg_gt right format, introduce back weights
        # elif self.seg_loss_mode== "only_bg":
        #     L_bg=bg_fg_prob_loss(pred_class_scores, class_gt, heatmaps,seg_gt)
        #     L_duc=depress_unexist_class_loss(pred_class_scores, class_gt, heatmaps,seg_gt)
        #     heatmap_loss=L_bg+L_duc

        else:
            raise NotImplementedError("This segmentation loss mode has not been implemented.")

        # classification loss
        # classification_loss=torch.nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)(pred_class_scores,class_gt.float()) # size (N, num_classes) flat tensor 
        weight = torch.ones(self.args.num_classes).to(self.device)
        weight[0] = 0
        classification_loss= F.multilabel_soft_margin_loss(pred_class_scores,class_gt.float(),weight=weight) # size (N, num_classes) flat tensor
        classification_loss*=self.args.batch_size

        # total loss
        # try adaptive weights:
        # hml_weight = 1.0
        # if self.alpha*heatmap_loss.item()>self.beta*classification_loss.item():
        #     hml_weight*=self.beta*classification_loss.item()/(self.alpha*heatmap_loss.item()+1)
        # t_loss=self.beta*classification_loss+self.alpha*hml_weight*heatmap_loss
        t_loss=self.beta*classification_loss+self.alpha*heatmap_loss
        return t_loss, self.alpha*heatmap_loss, self.beta*classification_loss

    def GlobalWeightedRankPooling(self,x,d=0.9):
        x,_=torch.sort(x.view(x.shape[0],x.shape[1],
                                        x.shape[2]*x.shape[3]),
                    dim=-1,descending=True)
        weights=torch.tensor([d ** i for i in range(x.shape[-1])]).type_as(x)
        weights=weights.unsqueeze(0).unsqueeze(0)
        x=torch.mul(x,weights)
        x=x.sum(-1)/weights.sum()
        return x

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
        elif method == "valley":
            mask_pred = F.softmax(torch.abs(adjusted_heatmap), 1)
        else:
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
            
           
            class_labels=class_labels.to(self.device)       
            data = data.to(self.device)
            
            # compute weights to correct imbalance bg/class_i TODO put into dedicated function
            # iter = (batch_idx+1)*len(data)
            # count_batch = torch.sum(class_labels,dim=0)
            # class_counts += count_batch
            # self.pos_weight = (iter-class_counts+1)/(class_counts+1)
            
            self.optimizer.zero_grad()
            class_scores= self.model(data,class_labels, only_classification=True)
            #loss=torch.nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)(class_scores,class_labels.float()) # size (N, num_classes) flat tensor 
            weight = torch.ones(self.args.num_classes).to(self.device)
            weight[0] = 0
            loss= F.multilabel_soft_margin_loss(class_scores,class_labels.float(),weight=weight)
            # loss*=self.args.pre_batch_size
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
        gen_log_message("train",epoch,train_loss , metrics,len(self.pre_train_loader.dataset),print_class_iou=False)
        # write_summaries(self.tb_writer1, train_loss , metrics,len(self.pre_train_loader.dataset), epoch+1)



    

    def pretraining_test(self,epoch):

        self.model.eval()
        test_loss = 0

        metrics=init_metrics(self.pretraining_test_metrics,self.args.num_classes)
        with torch.no_grad():
            for i, (data, class_labels,sem_gts,_) in enumerate(self.pre_test_loader):
       
                data = data.to(self.device)
                class_labels=class_labels.to(self.device)
                class_scores= self.model(data,class_labels, only_classification=True)
                
                # loss=torch.nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)(class_scores,class_labels.float()) # size (N, num_classes) flat tensor 
                weight = torch.ones(self.args.num_classes).to(self.device)
                weight[0] = 0
                loss= F.multilabel_soft_margin_loss(class_scores,class_labels.float(),weight=weight)
                # loss= F.multilabel_soft_margin_loss(class_scores,class_labels.float())
                # loss*=self.args.pre_batch_size
                test_loss += loss.item()
              
                metrics=update_metrics(metrics,class_scores,class_labels,None,None,sem_gts)
           
               
        test_loss /= len(self.pre_test_loader.dataset)

        if self.args.use_earlystopping:
            self.early_stopping(test_loss)

        _,_,_,_,match_ratio,avg_class_acc,_,_,_ = gen_log_message("test",epoch,test_loss, metrics,len(self.pre_test_loader.dataset),return_value=True,print_class_iou=False)
        # write_summaries(self.tb_writer2,test_loss,metrics, len(self.pre_test_loader.dataset),epoch) # check if this test_loader todo

        if self.args.wandb != 'None':
            wandb_utils.log_pretraining_test(self.seen_train_imgs,test_loss,match_ratio,avg_class_acc)  

        #save the model
        if self.args.save_interval is not None:
            if epoch%self.args.save_interval==0 or epoch==self.args.pre_epochs:
                self.save_model(epoch,"pre_train") 


    # --------------------------------------
    # Functions for offical train
    # --------------------------------------
        
    def train(self, epoch):

        self.model.train()
        train_loss = 0
        # class_counts = torch.zeros(self.args.num_classes).to(self.device) # TODO put into dedicated function - push to init
        metrics=init_metrics(self.train_metrics,self.args.num_classes)
        watch = wandb_utils.watch_gradients()

        # reshuffle sampler again and setup again trainloader
        self.train_loader=torch.utils.data.DataLoader(self.data.training_dataset,sampler=self.train_sampler.sample(),batch_size=self.args.batch_size)

        for batch_idx, (data, class_labels,sem_gts,sem_gt_exist) in enumerate(self.train_loader): 
            
            # measure number of samples before it goes to if function about self.args.only_send_labeled_data
            num_samples=len(data)
            # 1) prepare data
            sem_gts=sem_gts.long().to(self.device)
            class_labels=class_labels.to(self.device)
            data = data.to(self.device)

            # 1a) compute weights to correct imbalance bg/class_i TODO put into dedicated function
            # iter = (batch_idx+1)*len(data)
            # count_batch = torch.sum(class_labels,dim=0)
            # class_counts += count_batch
            # self.pos_weight = (iter-class_counts+1)/(class_counts+1)
            
            # 2) apply model on data and update gradients 
            self.optimizer.zero_grad()

            if self.args.iterative_gradients == True:
                class_scores = self.model(data,class_labels, only_classification=True)
                heatmaps_detached = self.model.forward_decoder_detached(class_scores)
                for class_idx in range(self.args.num_classes):
                    heatmaps = heatmaps_detached.clone() # N x classes x H x W
                    #heatmaps.requires_grad_() # setup possible graph for gradients
                    heatmap = self.model.forward_individual_class(class_scores,class_idx) # compute individual heatmap with gradients
                    heatmaps[:,class_idx] = heatmap
                    heatmaps=self.add_bg_heatmap(heatmaps) 
                    loss, loss1, loss2= self.loss_function(class_scores,class_labels,heatmaps,sem_gts,sem_gt_exist)
                    if loss1!=0:
                        loss1.backward(retain_graph=True) # backpropagate heatmap loss individually
                loss2.backward() # backpropagate classification loss
            else:
                if self.args.only_send_labeled_data:
                    data=data[sem_gt_exist==True]
                    class_labels=class_labels[sem_gt_exist==True]
                    sem_gts=sem_gts[sem_gt_exist==True]
                else:
                    pass
                class_scores,heatmaps = self.model(data,class_labels)
                loss, loss1, loss2= self.loss_function(class_scores,class_labels,heatmaps,sem_gts,sem_gt_exist)
                
                loss.backward()
                # print(f"iterative graidents==False, total loss {loss.item():.2f} ;h_loss {loss1.item():.2f} c_loss{loss2.item():.2f}")
            
            train_loss += loss.item()
            with torch.no_grad():
                if self.args.wandb != 'None':
                    watch.update(self.model,iter=self.seen_train_imgs)
            # TODO respective weight decay, is m0 on gpu??
            if self.args.pretrain_weight_decay > 0:
                loss_pre = torch.zeros(1,requires_grad=True,device='cuda:0')
                for param0,param in zip(self.model0.parameters(),self.model.parameters()):
                    loss_pre = loss_pre + torch.nn.MSELoss()(param,param0.detach())
                loss_pre = loss_pre * self.args.pretrain_weight_decay/2
                loss_pre.backward()
                print(loss_pre)

            self.optimizer.step()
            

            
            if self.args.clip>0:
                self.gradient_clip(self.model)
                   
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
                mask_scores,_,adjusted_heatmap = self.inferring_mask_scores(class_scores,heatmaps,method=self.args.mask_inference_method)

                # 5) compute and log metrics 
                metrics=update_metrics(metrics,class_scores,class_labels,adjusted_heatmap,mask_scores,sem_gts)

                # 6) check the confidence map
                if self.args.save_confidence_map and batch_idx==0:
                    pseudo_logits, pseudo_labels = torch.max(torch.softmax(heatmaps, dim=1), dim=1)
                    save_raw_confidencemap_sem_gt(4, 2,data[:8],pseudo_logits[:8],sem_gts[:8], f"{epoch}_{batch_idx}_train_confidence_map.png", 
                                    self.confidence_map_save_folder, color=False, normalize=self.args.normalize)
                

                if self.args.wandb != 'None':
                    wandb_utils.log_training(self.seen_train_imgs,epoch,batch_idx,num_samples,
                        loss2.item(),loss1.item(),metrics,
                        data,sem_gts,heatmaps,mask_scores,self.data.training_dataset.new_classes,
                        class_labels,class_scores,self.args.log_imgs,media_log_interval=3000)
        
                self.seen_train_imgs += num_samples
                

                remain=self.seen_train_imgs //3000
                if self.seen_train_imgs<remain*3000+self.args.batch_size:
                    self.test(epoch)


        train_loss=train_loss/ len(self.train_loader.dataset)
        gen_log_message("train",epoch,train_loss , metrics,len(self.train_loader.dataset),print_class_iou=False)
        # write_summaries(self.tb_writer1, train_loss , metrics,len(self.train_loader.dataset), epoch+1)
    
    def gradient_clip(self,model):
        for name, module in model.named_modules():
            if hasattr(module, 'weight'):
                torch.nn.utils.clip_grad_norm_(module.parameters(), self.args.clip)


        
    def test(self, epoch):
        
        self.model.eval()
        # init parameters etc. for tracking
        test_loss = 0
   
        metrics=init_metrics(self.test_metrics,self.args.num_classes)

        storing_examples = wandb_utils.storing_imgs(n_batches = len(self.data.testing_dataset),
                                                    test_batch_size=self.args.batch_size,seed=0)

        with torch.no_grad():
            for batch_idx, (data, class_labels,sem_gts,sem_gt_exists) in enumerate(self.test_loader):
                
                sem_gts=sem_gts.long().to(self.device)
                data = data.to(self.device)
                class_labels=class_labels.to(self.device)
                class_scores,heatmaps = self.model(data,class_labels)
                heatmaps=self.add_bg_heatmap(heatmaps) 
                tloss, test_loss1, test_loss2 = self.loss_function(class_scores, class_labels,heatmaps,sem_gts,sem_gt_exists)
                test_loss += tloss.item()

                # get masks scores based on predicted classes
                mask_scores,_,adjusted_heatmap = self.inferring_mask_scores(class_scores,heatmaps,method=self.args.mask_inference_method)
            
                # calculate the metrics and update# compute metrics 
                metrics=update_metrics(metrics,class_scores,class_labels,adjusted_heatmap,mask_scores,sem_gts)

                # average precision and accuracy
                # for i in range(sem_gts.shape[0]):
                #     ap = get_ap_scores(mask_scores[i:i+1], sem_gts[i:i+1])[0]
                #     acc, pix = pix_accuracy(mask_scores[i:i+1], sem_gts[i:i+1])
                #     acc_meter.update(acc, pix)
                #     ap_meter.update(ap)

                # save images and heatmaps for logging
                if self.args.wandb != 'None':
                    storing_examples.update(batch_idx, data.detach().cpu(),sem_gts.detach().cpu(),mask_scores)    

                # save confidence maps
                if self.args.save_confidence_map and batch_idx==0:
                    pseudo_logits, pseudo_labels = torch.max(torch.softmax(heatmaps, dim=1), dim=1)
                    save_raw_confidencemap_sem_gt(4, 2,data[:8],pseudo_logits[:8],sem_gts[:8], f"{epoch}_{batch_idx}_test_confidence_map.png", 
                                    self.confidence_map_save_folder, color=False, normalize=self.args.normalize)
                

                #save heatmaps
                # if i==30:
                #     print(f"      Test (random batch): test H-L {test_loss1.item()/len(data):.10f}; test C-L {test_loss2.item()/len(data):.4f}")
                #     save_raw_heatmap_sem_gt(4,2,data,heatmaps,sem_gts, f"{epoch}_test_heatmap.png" , self.heatmap_save_folder)


        test_loss /= len(self.test_loader.dataset)
        old_mIoU,old_iou_class,mIoU,iou_class,match_ratio,avg_class_acc,class_acc,pixel_acc,ap = gen_log_message("test",epoch,test_loss, metrics,len(self.test_loader.dataset),return_value=True,print_class_iou=True)
        # write_summaries(self.tb_writer2,test_loss,metrics,len(self.test_loader.dataset),epoch)

    
        if self.args.wandb != 'None':
            if self.args.log_tables and np.isscalar(class_acc)==False and np.isscalar(iou_class)==False:
                wandb_utils.log_testing_per_class_metrics(self.seen_train_imgs,"class_acc_table", self.class_acc_table, class_acc,self.table_count)
                wandb_utils.log_testing_per_class_metrics(self.seen_train_imgs,"class_iou_table", self.class_iou_table, iou_class,self.table_count)
                # wandb_utils.log_testing_per_class_metrics(self.seen_train_imgs,"class_old_iou_table", self.class_old_iou_table, old_iou_class,self.table_count)
                self.table_count+=1
            wandb_utils.log_testing(self.seen_train_imgs,
                                    test_loss,old_mIoU,mIoU,match_ratio,pixel_acc,ap,
                                    storing_examples.get_imgs(),storing_examples.get_gt_masks(),storing_examples.get_pred_masks(),self.data.training_dataset.new_classes)             
        #save the model
        # if self.args.save_interval is not None:
        #     if epoch%self.args.save_interval==0 or epoch==self.args.total_epochs:
        #         self.save_model(epoch,"offical_train") 

        if self.args.save_all_checkpoints:
            if self.args.pretrain_weight_name is not None:
                pretrained_model_name = self.args.pretrain_weight_name.split("/")[-1]
                pre_epochs=int(pretrained_model_name.split("_")[1])
            else:
                pre_epochs=0
            # save the model
            os.makedirs(f"{self.args.save_folder}"+"all_checkpoints/"+f"{self.args.backbone}_seed{self.args.seed}_pre_epochs{pre_epochs}",exist_ok=True)
            torch.save(self.model.state_dict(), 
                f"{self.args.save_folder}"+"all_checkpoints/"+f"{self.args.backbone}_seed{self.args.seed}_pre_epochs{pre_epochs}"+
                    f"/lrp0_{self.args.backbone}_iter{self.seen_train_imgs}_lab{self.args.n_masks}_lr{self.args.lr}_bs{self.args.batch_size}.pth")

        else:   
            if mIoU >= self.max_mIoU and self.args.save_folder is not None:
                # record the pret training epochs
                if self.args.pretrain_weight_name is not None:
                    pretrained_model_name = self.args.pretrain_weight_name.split("/")[-1]
                    pre_epochs=int(pretrained_model_name.split("_")[1])
                else:
                    pre_epochs=0
                # save the model
                os.makedirs(f"{self.args.save_folder}/{self.args.backbone}_seed{self.args.seed}_pre_epochs{pre_epochs}",exist_ok=True)
                torch.save(self.model.state_dict(), 
                    f"{self.args.save_folder}"+f"{self.args.backbone}_seed{self.args.seed}_pre_epochs{pre_epochs}"+
                        f"/lrp0_{self.args.backbone}_iter{self.seen_train_imgs}_lab{self.args.n_masks}_lr{self.args.lr}_bs{self.args.batch_size}.pth")
                
                # update mIoU
                try:
                    os.remove(f"{self.args.save_folder}"+f"{self.args.backbone}_seed{self.args.seed}_pre_epochs{pre_epochs}"+
                            f"/lrp0_{self.args.backbone}_iter{self.max_mIoU_iter}_lab{self.args.n_masks}_lr{self.args.lr}_bs{self.args.batch_size}.pth")
                except FileNotFoundError:
                    pass
                self.max_mIoU= mIoU
                self.max_mIoU_iter=self.seen_train_imgs
        
        self.model.train()

        
 





