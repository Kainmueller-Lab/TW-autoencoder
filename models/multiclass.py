'''
The multi-class classification case, either fully supervised or weakly supervised
'''

import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
import sys
sys.path.append('../')

from arch.architectures_utils import FwdHooks


from datasets import MNIST, EMNIST, FashionMNIST
import numpy as  np
from torch.autograd import Variable
import os
from utils.plot_utils import *
from utils.metrics_utils import *

# from arch.arch_MNIST import  CNN_Decoder,CNN_Encoder # encoder and decoder for MNIST dataset




# --------------------------------------
# Network for multi-class classification
# --------------------------------------
class Network(nn.Module):
    def __init__(self, args):
        super(Network, self).__init__()
        output_size = args.embedding_size
        self.num_classes=args.num_classes
        self._init_params(args)

        CNN_Encoder,CNN_Decoder=self.import_arch(args.backbone)
        self.encoder = CNN_Encoder(output_size,num_classes=args.num_classes)
        self.register_hooks(self.encoder)
        
        # run test_forward for assigning the attribute of in_tensor.shape 
        # for the transposeconv_2d layer in the decoder
        self.test_forward(self.encoder, self.input_size)
        self.tied_weights_decoder = CNN_Decoder(self.encoder)

    def import_arch(self, keyword):
        if keyword=="CNN_MNIST":
            from arch.arch_MNIST import  CNN_Decoder,CNN_Encoder # encoder and decoder for MNIST dataset
        elif keyword=="vgg":
            from arch.arch_vgg import  CNN_Decoder,CNN_Encoder # encoder and decoder for MNIST dataset
        elif keyword=="efficientnet":
            from arch.arch_efficientnet import  CNN_Decoder,CNN_Encoder # encoder and decoder for MNIST dataset
        return CNN_Encoder,CNN_Decoder

    def _init_params(self,args):
        if args.dataset in ['MNIST','EMNIST','FashionMNIST']:
            self.img_channel=1
            self.input_size=(1,28,28)
        elif args.dataset== 'Lizard':
            self.img_channel=3
            self.input_size=(3,256,256)
        else:
            print("Dataset not supported")


    def forward(self, x, targets=None):
        '''
        targets: torch.tensor (N,)
        '''
        z = self.encoder(x)
        gt_heatmap=self.tied_weights_decoder(self.indi_features(z,targets))

        # pred_heatmap=self.tied_weights_decoder(self.indi_features(z,torch.max(z,1)[1]))
        pred_heatmap=self.tied_weights_decoder(self.indi_features(z,torch.ones(z.shape[0]).long()))
        return z, gt_heatmap,pred_heatmap

    def test_forward(self,module,input_size):
        x=Variable(torch.ones(2, *input_size)) # bacth_size must larger than 1 for batchnorm1d   
        y=module(x)     
        return

    def register_hooks(self, parent_module):
        fwd_hooks=FwdHooks()
        for mod in parent_module.children():
            print("LRP processing module... ", mod)     
            if list(mod.children()):
                self.register_hooks(mod)
                continue
            mod.register_forward_hook(
                fwd_hooks.get_layer_fwd_hook(mod))


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
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # device=class_scores.device todo
        N=class_scores.shape[0]
        
        if init_type == "softmax":
            T=self.softmaxlayer_lrp(class_scores,targets,device)

        elif init_type == "normal":
            T = torch.zeros(class_scores.shape).to(device)
            T[range(N),targets]=torch.abs(class_scores[range(N),targets].detach()) #check len(targets)=1
            # T[range(N),targets]=class_scores[range(N),targets].detach()

        elif init_type== "contrastive":
            T=torch.abs(class_scores.detach().to(device))
            T[range(N),targets]=0.0

        elif init_type == "normal-contrastive":
            T = torch.ones(class_scores.shape).to(device)*(-1)
            T[range(N),targets]=1.0
            T=T*class_scores.detach()
  
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
        targets=torch.LongTensor(targets).to(device)
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
        self.train_loader = self.data.train_loader
        self.test_loader = self.data.test_loader

        
        self.model = Network(args)
        
        self.model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4)
        self.alpha=5e-8
        self.beta=0

        # heatmap save folder
        self.heatmap_save_folder=f"results/heatmaps/{self.args.dataset}_{self.args.num_classes}/alpha_{self.alpha}_beta_{self.beta}/"
        os.makedirs(self.heatmap_save_folder,exist_ok=True)
        # TODO: add tensorboard 
  
    def _init_dataset(self):
        if self.args.dataset == 'MNIST':
            self.data = MNIST(self.args)
        elif self.args.dataset == 'EMNIST':
            self.data = EMNIST(self.args)
        elif self.args.dataset == 'FashionMNIST':
            self.data = FashionMNIST(self.args)
        # todo add lizard dataset
        else:
            print("Dataset not supported")
           

    def loss_function(self, pred_class_scores, class_gt, gt_label_heatmap,seg_gt):
        weight_map=(1-seg_gt)*torch.ones_like(gt_label_heatmap).to(self.device)
        # weight_map=torch.ones_like(pred_heatmap).to(self.device)

        heatmap_loss=nn.BCEWithLogitsLoss(weight=weight_map,reduction="sum")(input=gt_label_heatmap,target=seg_gt)
        classification_loss=nn.CrossEntropyLoss(reduction="sum")(pred_class_scores,class_gt.long()) # size (80)
        t_loss=self.alpha*heatmap_loss+self.beta*classification_loss
        return t_loss, self.alpha*heatmap_loss, self.beta*classification_loss

    def train(self, epoch):
        self.model.train()
        train_loss = 0
        metrics={'num_correct':0,
            'num_samples':0,
            'pixel_acc':0}
        for batch_idx, (data, class_labels) in enumerate(self.train_loader):
           
            # generate sem_gt from data
            sem_gts=generate_sem_gt(data,0.1).to(self.device)
           
            #normalize the input data
            # where "-1.0" corresponds to black and "+1.0" corresponds to white.
            data=(data-0.5)*2
            class_labels=class_labels.to(self.device)
            data = data.to(self.device)
            self.optimizer.zero_grad()
            class_scores,gt_label_heatmap,pred_label_heatmap = self.model(data,class_labels)
            loss, loss1,loss2= self.loss_function(class_scores, class_labels, gt_label_heatmap,sem_gts)
            loss.backward()
            train_loss += loss.item()
            self.optimizer.step()
            metrics=get_accuracy("classification",metrics, class_scores,class_labels)
            metrics=get_accuracy("segmentation",metrics,gt_label_heatmap,sem_gts)
            if batch_idx==5:
                print(f"====> Epoch: {epoch} Train (random batch): heatmap loss {loss1.item()/len(data):.10f}; classification loss {loss2.item()/len(data):.4f}")
                pred_labels=torch.max(class_scores,1)[1]
                corrects= (class_labels.long()==pred_labels)
                save_heatmap_sem_gt(24,8, gt_label_heatmap,pred_label_heatmap , sem_gts,corrects, f"{epoch}_train_heatmap.png",self.heatmap_save_folder)

        print(f'====> Epoch: {epoch} Train average loss: {train_loss / len(self.train_loader.dataset):.4f}; ' + 
                f'classification accuracy :{metrics["num_correct"]/metrics["num_samples"]: .4f}({metrics["num_correct"]}/{metrics["num_samples"]}); '+
                f'heatmap accuracy: {metrics["pixel_acc"]/len(self.train_loader.dataset):.4f}')

    def test(self, epoch):
        self.model.eval()
        test_loss = 0
        metrics={'num_correct':0,
            'num_samples':0,
            'pixel_acc':0}
        with torch.no_grad():
            for i, (data, class_labels) in enumerate(self.test_loader):
                sem_gts=generate_sem_gt(data,0.1).to(self.device)
                data=(data-0.5)*2
                data = data.to(self.device)
                class_labels=class_labels.to(self.device)
                class_scores,gt_label_heatmap,pred_label_heatmap = self.model(data,class_labels)
                tloss, test_loss1, test_loss2 = self.loss_function(class_scores, class_labels, gt_label_heatmap,sem_gts)
                test_loss+=tloss.item()
                # accuracy on classification and segmentation
                metrics=get_accuracy("classification",metrics, class_scores,class_labels)
                metrics=get_accuracy("segmentation",metrics,gt_label_heatmap,sem_gts)

                #save heatmaps
                if i==5:
                    print(f"      Test (random batch): test H-L {test_loss1.item()/len(data):.10f}; test C-L {test_loss2.item()/len(data):.4f}")
                    pred_labels=torch.max(class_scores,1)[1]
                    corrects= (class_labels.long()==pred_labels)
                    save_heatmap_sem_gt(24,8,gt_label_heatmap,pred_label_heatmap , sem_gts,corrects, f"{epoch}_test_heatmap.png",self.heatmap_save_folder)


        test_loss /= len(self.test_loader.dataset)
        print(f'      Test average loss: {test_loss:.4f} ' +
            f'test C-A :{metrics["num_correct"]/metrics["num_samples"]: .4f}({metrics["num_correct"]}/{metrics["num_samples"]}); '+
            f'test H-A: {metrics["pixel_acc"]/len(self.test_loader.dataset):.4f}')
        print('------'*6)

def generate_sem_gt(mnist_data,threshold):
    #the segmentation gt masks should have dtype = torch.FloatTensor and same size as data
    return (mnist_data>threshold).float() 