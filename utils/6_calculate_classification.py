import argparse, os, sys
import numpy as np
import torch
from datetime import datetime
import wandb
import random
sys.path.append("..")  # Adds the parent directory to the sys.path
from baseline_arch.multi_task_unet import MTUNet
from models.multilabel import Network
from datasets import PASCAL_dataset
from matplotlib.colors import to_rgb
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import metrics

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
def fill_in_columns_name(dict, name_list):
    for n_masks in dict['list_n_masks']:
        name_list.append(f"{dict['name']}_{n_masks}")
    return name_list

# https://www.kaggle.com/code/kmkarakaya/multi-label-model-evaluation
def my_iou(outputs,labels, class_labels):
    # this is the modified version of get_iou function in metric_utils.py
    num_classes=outputs.shape[1]
    outputs = torch.stack([ torch.argmax(outputs,dim=1)==i for i in range(num_classes) ],dim=1)
    labels = torch.stack([ labels==i for i in range(num_classes) ],dim=1)
    intersection=torch.sum(torch.logical_and(outputs,labels), axis=(2,3))
    union=torch.sum(torch.logical_or(outputs,labels), axis=(2,3))
    iou = intersection  / (union + 1e-8) # n_samples x classes
    num_classes_per_sample=torch.count_nonzero(class_labels, dim=1) # n_samples
    iou = torch.sum(iou,dim=1) /num_classes_per_sample # n_samples
    return iou

def eval_classification(class_scores, class_labels,model_name, n_masks, metrics_list):
    class_scores=class_scores[:,1:]
    class_labels=class_labels[:,1:]
    for me in metrics_list:
        if me=="F1_score":
            ########calculate the F1 score
            # f1_score=metrics.f1_score(class_labels, class_scores, average=None)
            # print(f"The micro F1 score {f1_score.mean()} is for model{model_name}_labe{n_masks}.")
            # f1_score=metrics.f1_score(class_labels, class_scores, average='micro')
            # print(f"The micro F1 score {f1_score:.2f} is for model{model_name}_labe{n_masks}.")
            f1_score=metrics.f1_score(class_labels, class_scores, average='macro')
            print(f"The macro(unweighted) F1 score {f1_score:.4f} is for model{model_name}_labe{n_masks}.")
            f1_score=metrics.f1_score(class_labels, class_scores, average='weighted')
            print(f"The weighted F1 score {f1_score:.4f} is for model{model_name}_labe{n_masks}.")
            # f1_score=metrics.f1_score(class_labels, class_scores, average='samples')
            # print(f"The samples F1 score {f1_score:.2f} is for model{model_name}_labe{n_masks}.")
        elif me=="mAP":
            ############# mean average precision
            print(f"The mAP is for model{model_name}_labe{n_masks}.")
        elif me=="avg_accuracy":
             ############# mean average precision
            print(f"The mAP is for model{model_name}_labe{n_masks}.")
            
    return
            

def intialize_model(args, dict):
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
    return model            
    

def load_checkpoint(args, model,name,iteration,seed, n_masks):
    checkpoint_name=f"{args.checkpoint_folder}"+f"{args.backbone}_seed{seed}_pre_epochs{args.pre_epochs}" + \
        f"/{name}_{args.backbone}_iter{iteration}_lab{n_masks}_lr{args.lr}_bs{args.batch_size}.pth"
    checkpoint=torch.load(checkpoint_name)
    model.load_state_dict(checkpoint)
    return model

## go through for every model and record the miou for each test image
def get_classification_performance_for_dict(args,test_loader,model_name,n_masks,metrics):
    ## iter through the test data and calculate the miou
    with torch.no_grad(): # must add this line of code
        list_class_scores=[]
        list_class_labels=[] 
        for batch_idx, (data, class_labels, sem_gts, sem_gt_exists) in tqdm(enumerate(test_loader)):
            sem_gts = sem_gts.long().to(device)
            data = data.to(device)
            class_labels = class_labels.to(device)


            # predict class_scores
            if model_name=="std_unet":
                raise KeyError
            elif model_name=="mt_unet":
                _,class_scores = model(data)
            else:
                class_scores=model(data, only_classification=True)

            #calculate the classification performance

            class_scores=(class_scores>0).int()

            list_class_scores.append(class_scores.detach().cpu())
            list_class_labels.append(class_labels.detach().cpu())


        t_class_scores=torch.cat(list_class_scores,dim=0)
        t_class_labels=torch.cat(list_class_labels,dim=0)


        eval_classification(t_class_scores,t_class_labels,model_name,n_masks, metrics)
    print(f"Finish for the model {model_name}")

    return 

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
      
        # for backbone resnet50 seed42
        # lrp0_dict={
        #     'name':'lrp0',
        #     'seed':42,
        #     'list_iter':[198000,249000,246000,297000], #change TODO
        #     'list_n_masks':[20,100,500,None],
        #     'test_batch_size':10
        # }
        # lrp0_dict={
        #     'name':'lrp0',
        #     'seed':41,
        #     'list_iter':[198000,195000,279000,393000], #change TODO
        #     'list_n_masks':[20,100,500,None],
        #     'test_batch_size':10
        # }
        lrp0_dict={
            'name':'lrp0',
            'seed':40,
            'list_iter':[198000,198000,270000,282000], #change TODO
            'list_n_masks':[20,100,500,None],
            'test_batch_size':10
        }


        # # for backbone vgg16bn seed42
        # lrp0_dict={
        #     'name':'lrp0',
        #     'seed': 41,
        #     'list_iter':[138008,183008,288008,351008], #change TODO
        #     'list_n_masks':[20,100,500,None],
        #     'test_batch_size':10
        # }

        # lrp0_dict={
        #     'name':'lrp0',
        #     'seed': 40,
        #     'list_iter':[141008,177008,219008,327008], #change TODO
        #     'list_n_masks':[20,100,500,None],
        #     'test_batch_size':10
        # }

        
        ## load test dataset according to args
        print("-----------load test dataset-----------------")
        random.seed(100)
        prefixed_args={"path": args.data_path, 
                        "seed": args.seed,
                        "n_masks": None,
                        "uniform_masks": args.uniform_masks,
                        "reduce_classes":args.reduce_classes,
                        "weighted_sampling": args.weighted_mask_sampling,
                        "normalize": args.normalize,
                        "trafo_mode": args.trafo_mode
                        }

    
        
        #about the dataset
        testing_dataset = PASCAL_dataset(**prefixed_args,mode="test",full_supervison=True)
        test_loader = torch.utils.data.DataLoader(testing_dataset, batch_size=40)
        class_dict=testing_dataset.new_classes
    
    
        metrics_list=["F1_score"]


        model=intialize_model(args, lrp0_dict)
        #load the pretrain checkpoint
        if args.backbone=="vgg16_bn":
            checkpoint_name=f"../snapshot/"+ f"vggbn_8_pre_train_21.pth"
        elif args.backbone=="resnet50":
            checkpoint_name=f"../snapshot/"+ f"{args.backbone}_{args.pre_epochs}_pre_train_21.pth"
        elif args.backbone=="vgg16":
            checkpoint_name=f"../snapshot/"+ f"vgg_10_pre_train_21.pth"

        # checkpoint_name=f"../snapshot/"+ f"{args.backbone}_{args.pre_epochs}_pre_train_21.pth"
        
        model.to(device)
        model.eval()
        #### TODO check
        ###### why you need to fist set eval mode and then load the pretrain weights
        with torch.no_grad():
            print(f"Load the checkpint for iteration --------------------156760")
          
            checkpoint=torch.load(checkpoint_name)
            model.load_state_dict(checkpoint)
            print(f"re-Load the checkpint for iteration----------------------- 156760")
            get_classification_performance_for_dict(args,test_loader,lrp0_dict['name'],0, metrics_list)

#             # for iteration, n_masks in zip(lrp0_dict['list_iter'],lrp0_dict['list_n_masks']):
#                 model=load_checkpoint(args, model,lrp0_dict['name'], iteration, lrp0_dict['seed'],n_masks)
#                 get_classification_performance_for_dict(args,test_loader,lrp0_dict['name'],n_masks, metrics_list)
# #          
            # iteration=195000
            # args.checkpoint_folder="../snapshot/lcal0/all_checkpoints/"
            # args.backbone="resnet50"
            # args.seed=42
            # args.pre_epochs=10
            # n_masks=100
            # for iteration in np.arange(159000,207000+1,3000):
            #     model=load_checkpoint(args, model,lrp0_dict['name'], iteration, args.seed,n_masks)
            #     print(f"The following is for iteration--------------------{iteration}.")
            #     get_classification_performance_for_dict(args,test_loader,lrp0_dict['name'],n_masks, metrics_list)

