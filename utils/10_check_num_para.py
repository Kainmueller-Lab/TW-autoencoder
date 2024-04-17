import argparse, os, sys
import numpy as np
import torch
import torch.nn as nn
from datetime import datetime
import wandb
import random
sys.path.append("..")  # Adds the parent directory to the sys.path
from baseline_arch.multi_task_unet import MTUNet
from models.multilabel import Network
from datasets import PASCAL_dataset
from matplotlib.colors import to_rgb
from PIL import Image



def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def str_or_none(v):
    if v is None:
        return v
    if v.lower() in ('none', 'no', 'false', 'f', 'n', '0'):
        return None
    else:
        return v
    
def int_or_none(v):
    if v is None:
        return v
    if v.lower() in ('none', 'no', 'false', 'f', 'n'):
        return None
    else:
        return int(v)


## needed function


def load_checkpoint(args, model,name,iteration, n_masks):
    checkpoint_name=f"{args.checkpoint_folder}"+f"{args.backbone}_seed{args.seed}_pre_epochs{args.pre_epochs}" + \
        f"/{name}_{args.backbone}_iter{iteration}_lab{n_masks}_lr{args.lr}_bs{args.batch_size}.pth"
    checkpoint=torch.load(checkpoint_name)
    model.load_state_dict(checkpoint)
    return model

## go through for every model and record the miou for each test image
def count_para(args,dict):
    ## import model
    print(f"-----------import model {dict['name']}-----------------")
    if dict['name'] == 'std_unet':
        args.add_classification=False
        model = MTUNet(3, args.backbone,  args.num_classes, args.add_classification)
    elif dict['name'] == 'mt_unet':
        args.add_classification=True
        model = MTUNet(3, args.backbone,  args.num_classes, args.add_classification)
    elif dict['name'] == 'lrp0':
        args.add_classification = True
        args.use_old_xai = False
        args.xai = 'LRP_epsilon'
        args.epsilon = 1e-8
        args.alpha = 1
        args.memory_efficient = False
        args.detach_bias = True
        model = Network(args)
    else:
        raise NotImplementedError
    model.to(device)


    list_param=[]

    for name, p in model.named_parameters():
        if p.requires_grad:
            print(name, p.numel())
            list_param.append(p.numel())

    print(sum(list_param),"manual sum")

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"The numbder of total parameters is {pytorch_total_params}")
    return 
    # # pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # return


class LRP_linear(nn.Module):
    def __init__(self, linear, **kwargs):
        super(LRP_linear, self).__init__()
        assert isinstance(linear,nn.Linear), "Please tie with a linear layer"
        self.m=linear       
        self.inv_m=nn.Linear(in_features=self.m.out_features, out_features=self.m.in_features, bias=None) #bias=None
        self.inv_m.weight=nn.Parameter(self.m.weight.t())

    def forward(self,relevance_in):
        relevance_in=self.inv_m(relevance_in)
        return relevance_in
     
if __name__ == "__main__":

    parser = argparse.ArgumentParser(
            description='Main function to call training for different AutoEncoders')

    # dataset
    parser.add_argument('--dataset', type=str, default='PASCAL',choices=['MNIST','Lizard','PASCAL','FashionMNIST'], help='Which dataset to use') 
    parser.add_argument('--data_path',type=str,default="/home/xiaoyan/Documents/Data/", metavar='N', help='setup the origin of the data')
    parser.add_argument('--num-classes', type=int, default=21, metavar='N', help='reduce classes if this is implemented in dataset')
    parser.add_argument('--reduce-classes', type=str2bool, default=False, metavar='S', help='reduce classes in dataset if this is implemented in dataset / note num classes must align')
    parser.add_argument('--seed', type=int, default=42, metavar='S', help='random seed (default: 1)')
    parser.add_argument('--uniform-class-sampling', type=str2bool, default=True, metavar='S', help='if use uniform class sampling strategy')
    parser.add_argument('--weighted-mask-sampling', type=bool, default=True, metavar='N', help='probability of sampling a mask') # outdated
    parser.add_argument('--prob-sample-mask', type=float, default=0.5, metavar='N', help='probability of sampling a mask')
    parser.add_argument('--fluct-masks-batch', type=float, default=1.0, metavar='N', help='fluctuation of number of images with mask per batch')
    parser.add_argument('--normalize', type=str2bool, default=False, metavar='N', help='if normalize the input tensor')  
    parser.add_argument('--uniform-masks', type=float, default=0.0, metavar='N', help='switches the distribution of sampled masks to uniform distribution regarding the class labels')
    parser.add_argument('--trafo-mode', type=int, default=0, metavar='N', help='different types of augmentations')

    # model
    parser.add_argument('--name', type=str, default="std_unet",choices=['std_unet','mt_unet','lrp0'])
    parser.add_argument('--backbone', type=str, default='resnet50',choices=['vgg16','vgg16_bn','resnet18','resnet34','resnet50','resnet101'], help='the backbone for the encoder')
    parser.add_argument('--pre_epochs', type=int, default=10,help="related to loaded checkpoint name")
    parser.add_argument('--no-cuda', action='store_true', default=False, help='enables CUDA training')

    # train
    parser.add_argument('--batch-size', type=int, default=10, help="related to loaded checkpoint name")
    parser.add_argument('--checkpoint_folder', type=str, default='../snapshot/', help='examples:"snapshot/", "/fast/AG_Kainmueller/xyu/"') #checkpoint_folder should be "snapshot/" or "/fast/AG_Kainmueller/xyu/"

    # loss desgin
    parser.add_argument('--lr', type=float, default=1e-5, metavar='N', help='learning rate')
    parser.add_argument('--cuda-dev', type=str, default='cuda', metavar='N')
    
    # plot
    parser.add_argument('--order_by', type=str, default='head',choices=['head','tail'])
    parser.add_argument('--function', type=str, default='generate',choices=['generate','check'])
    

    args = parser.parse_args()

    if args.function=="generate":
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)

        args.cuda = not args.no_cuda and torch.cuda.is_available()

        device = torch.device(args.cuda_dev if args.cuda else "cpu")

        
        ## manual input information for the checkpoints
        std_unet_dict={
            'name':'std_unet',
            'list_iter':[202500] , #change TODO
            'list_n_masks':[20],
            'test_batch_size':50
        }
        mt_unet_dict={
            'name':'mt_unet',
            'list_iter':[207000], #change TODO
            'list_n_masks':[20],
            'test_batch_size':50
        }
        lrp0_dict={
            'name':'lrp0',
            'list_iter':[198000], #change TODO
            'list_n_masks':[20],
            'test_batch_size':10
        }


  
            

        count_para(args,std_unet_dict)
        # count_para(args,mt_unet_dict)
        # count_para(args,lrp0_dict)



        # test for bounding parameters cases
        # a= torch.nn.Linear(20, 30)
        # b= LRP_linear(a)
        # b.weight= torch.nn.Parameter(a.weight.t())
        # model=torch.nn.Sequential(a,b)
        # pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        # pytorch_total_params = sum(p.numel() for p in a.parameters() if p.requires_grad)
        # print(pytorch_total_params)
 