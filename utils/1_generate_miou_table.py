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



def load_checkpoint(args, model,name,iteration, n_masks):
    checkpoint_name=f"{args.checkpoint_folder}"+f"{args.backbone}_seed{args.seed}_pre_epochs{args.pre_epochs}" + \
        f"/{name}_{args.backbone}_iter{iteration}_lab{n_masks}_lr{args.lr}_bs{args.batch_size}.pth"
    checkpoint=torch.load(checkpoint_name)
    model.load_state_dict(checkpoint)
    return model

## go through for every model and record the miou for each test image
def get_miou_for_dict(args,testing_dataset,dict, df):
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

    with torch.no_grad(): # must add this line of code
        ## iter through the test data and calculate the miou
        for iteration, n_masks in zip(dict['list_iter'],dict['list_n_masks']):
            model=load_checkpoint(args, model, dict['name'], iteration, n_masks)
            model.to(device)
            model.eval()

            test_miou_list=[]
            test_loader = torch.utils.data.DataLoader(testing_dataset, batch_size=dict['test_batch_size'])
            for batch_idx, (data, class_labels, sem_gts, sem_gt_exists) in tqdm(enumerate(test_loader)):
                sem_gts = sem_gts.long().to(device)
                data = data.to(device)
                class_labels = class_labels.to(device)

                # predict heatmap
                if dict['name']=="std_unet":
                    heatmaps = model(data)
                elif dict['name']=="mt_unet":
                    heatmaps,_ = model(data)
                else:
                    _ ,heatmaps= model(data)

                iou=my_iou(heatmaps.detach().cpu(),sem_gts.detach().cpu(), class_labels.detach().cpu()) # N
                # img_idx=batch_idx*test_batch_size+range(1,len(data)+1,1) 
                test_miou_list.append(iou)

            test_miou=torch.cat(test_miou_list, dim=0) # can not use torch.stack here
            print(len(list(test_miou)),"len of test miou")
            df[f"{dict['name']}_{n_masks}"]=list(test_miou)

    print(f"Finish for the model {dict['name']}")

    return df

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
            'list_iter':[202500,252000,259500,304500] , #change TODO
            'list_n_masks':[20,100,500,None],
            'test_batch_size':50
        }
        mt_unet_dict={
            'name':'mt_unet',
            'list_iter':[207000,285000,432000,483000], #change TODO
            'list_n_masks':[20,100,500,None],
            'test_batch_size':50
        }
        lrp0_dict={
            'name':'lrp0',
            'list_iter':[198000,249000,246000,285000], #change TODO
            'list_n_masks':[20,100,500,None],
            'test_batch_size':10
        }

        if not os.path.exists("../results/csv/"):
            os.makedirs("../results/csv/")
        ## create a empty dataframe
        print("-----------initialize dataframe-----------------")
        columns_name_list=['img_idx']
        columns_name_list=fill_in_columns_name(std_unet_dict,columns_name_list)
        columns_name_list=fill_in_columns_name(mt_unet_dict,columns_name_list)
        columns_name_list=fill_in_columns_name(lrp0_dict,columns_name_list)

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

        testing_dataset = PASCAL_dataset(**prefixed_args,mode="test",full_supervison=True)
        
        # df=pd.DataFrame(index=range(0,len(testing_dataset),1),columns=columns_name_list) # no need to give index argument
        df=pd.DataFrame(columns=columns_name_list)
        
        df['img_idx']=range(0,len(testing_dataset),1)

        df=get_miou_for_dict(args,testing_dataset, std_unet_dict, df)
        df=get_miou_for_dict(args,testing_dataset, mt_unet_dict, df)
        df=get_miou_for_dict(args,testing_dataset, lrp0_dict, df)

        file_name = f"{args.name}_{args.backbone}_lr{args.lr}_bs{args.batch_size}.csv"
        path_save = "../results/csv/" + file_name  
        print(f"Created csv file will be save in the following path {path_save}")
     

        

        # calculat the difference between columns
        df['lrp0_20-std_unet_20']=df['lrp0_20']-df['std_unet_20']
        df['lrp0_20-mt_unet_20']=df['lrp0_20']-df['mt_unet_20']
        df['lrp0_100-std_unet_100']=df['lrp0_100']-df['std_unet_100']
        df['lrp0_100-mt_unet_100']=df['lrp0_100']-df['mt_unet_100']

        df.to_csv(path_save)
    
    else:
    
        file_name = f"{args.name}_{args.backbone}_lr{args.lr}_bs{args.batch_size}.csv"
        path_save = "../results/csv/" + file_name 
        df = pd.read_csv(path_save)
        # sort the dataframe 
        num_samples=20
        if args.order_by == "tail":
            a=df.sort_values('lrp0_20-std_unet_20')['img_idx'].tail(num_samples)
            print(f"The last {num_samples} samples for the difference between lrp0 and UNet (20 labeled) is {a.tolist()}")
            b=df.sort_values('lrp0_20-mt_unet_20')['img_idx'].tail(num_samples)
            print(f"The last {num_samples} samples for the difference between lrp0 and Multi-task UNet (20 labeled) is {b.tolist()}")
            c=df.sort_values('lrp0_100-std_unet_100')['img_idx'].tail(num_samples)
            print(f"The last {num_samples} samples for the difference between lrp0 and UNet (100 labeled) is {c.tolist()}")
            d=df.sort_values('lrp0_100-mt_unet_100')['img_idx'].tail(num_samples)
            print(f"The last {num_samples} samples for the difference between lrp0 and Multi-task UNet (100 labeled) is {d.tolist()}")
        else:
            a=df.sort_values('lrp0_20-std_unet_20')['img_idx'].head(num_samples)
            print(f"The fist {num_samples} samples for the difference between lrp0 and UNet (20 labeled) is {a.tolist()}")
            b=df.sort_values('lrp0_20-mt_unet_20')['img_idx'].head(num_samples)
            print(f"The fist {num_samples} samples for the difference between lrp0 and Multi-task UNet (20 labeled) is {b.tolist()}")
            c=df.sort_values('lrp0_100-std_unet_100')['img_idx'].head(num_samples)
            print(f"The fist {num_samples} samples for the difference between lrp0 and UNet (100 labeled) is {c.tolist()}")
            d=df.sort_values('lrp0_100-mt_unet_100')['img_idx'].head(num_samples)
            print(f"The fist {num_samples} samples for the difference between lrp0 and Multi-task UNet (100 labeled) is {d.tolist()}")




    