import argparse, os, sys
import numpy as np
import torch
from datetime import datetime
import wandb
import random

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

parser = argparse.ArgumentParser(
        description='Main function to call training for different AutoEncoders')
# dataset
parser.add_argument('--dataset', type=str, default='PASCAL',choices=['MNIST','Lizard','PASCAL','FashionMNIST'],
                    help='Which dataset to use') 
parser.add_argument('--data_path',type=str,default="/home/xiaoyan/Documents/Data/",metavar='N',
                    help='setup the origin of the data')
parser.add_argument('--num-classes', type=int, default=21, metavar='N',
                    help='reduce classes if this is implemented in dataset')
parser.add_argument('--reduce-classes', type=str2bool, default=False, metavar='S',
                    help='reduce classes in dataset if this is implemented in dataset / note num classes must align')
parser.add_argument('--seed', type=int, default=42, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--uniform-class-sampling', type=str2bool, default=True, metavar='S',
                    help='if use uniform class sampling strategy')
parser.add_argument('--weighted-mask-sampling', type=bool, default=True, metavar='N',
            help='probability of sampling a mask')
parser.add_argument('--prob-sample-mask', type=float, default=0.5, metavar='N',
            help='probability of sampling a mask')
parser.add_argument('--fluct-masks-batch', type=float, default=1.0, metavar='N',
            help='fluctuation of number of images with mask per batch')
parser.add_argument('--normalize', type=str2bool, default=False, metavar='N',
                    help='if normalize the input tensor')     
parser.add_argument('--n-masks', type=int_or_none, default=None, metavar='N',
                    help='number of masks applied during training of the original training masks')
parser.add_argument('--uniform-masks', type=float, default=0.0, metavar='N',
                    help='switches the distribution of sampled masks to uniform distribution regarding the class labels')     
parser.add_argument('--trafo-mode', type=int, default=0, metavar='N',
                    help='different types of augmentations')
       
# model
parser.add_argument('--model', type=str, default='multi_task_unet',choices=['unet','multi_task_unet'],
                    help='Which architecture to use')                   
parser.add_argument('--backbone', type=str, default='vgg16',choices=['vgg16','vgg16_bn','resnet18','resnet34','resnet50','resnet101'],
                    help='the backbone for the encoder')  #detail see documents baseline_multilabel
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')

# pretrain
parser.add_argument('--pretrain-weight-name', type=str_or_none, default=None, metavar='N',
                    help='the path to save the pretrain weights')   # example : './snapshot/vgg_12_pre_train'
parser.add_argument('--pre-batch-size', type=int, default=None, metavar='N',
                    help='input batch size for training (default: 32)')
parser.add_argument('--pre-epochs', type=int, default=10, metavar='N',
                    help='number of pre-training epochs which only do classification')
parser.add_argument('--use-earlystopping',type=str2bool,default=False,metavar='N', help='use earlier stopping')
parser.add_argument('--save-interval', type=int_or_none, default=None, metavar='N') #save_interval both exist in train and pretrain

# train
parser.add_argument('--batch-size', type=int, default=40, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=100, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--results_path', type=str, default='results/', metavar='N', help='Where to store images')
parser.add_argument('--wandb', type=str, default='None', metavar='N', help='Sets up a wandb project')
# if log imgs to wandb during training
parser.add_argument('--log_imgs', type=str2bool, default=True)
parser.add_argument('--save_folder', type=str, default=None, help='examples:"snapshot/", "/fast/AG_Kainmueller/xyu/"') #save_folder should be "snapshot/" or "/fast/AG_Kainmueller/xyu/"
parser.add_argument('--save_all_checkpoints', type=str2bool, default=False) # for heatmaps evolution figures
## arguments for computational performance
parser.add_argument('--iterative-gradients', type=str2bool, default=False, metavar='S',
                    help='choice between computing the gradients iteratively for each class-related heatmap (memory efficient but more backwards passes) / or in one pass')

## loss desgin
parser.add_argument('--add_classification', type=str2bool, default=False)
parser.add_argument('--loss_impact_seg', type=float, default=0.1, metavar='N',
                    help='scaling / impact of the segmentation loss')
parser.add_argument('--loss_impact_bottleneck', type=float, default=1, metavar='N',
                    help='scaling / impact of the bottleneck loss')            
parser.add_argument('--bg_weight_in_BCE_hm_loss', type=float, default=0, metavar='N',
                    help='weight for class 0 in BCE loss')
parser.add_argument('--nonexist_weight_in_BCE_hm_loss', type=float, default=0.1, metavar='N',
                    help='weight for non-existing classes (per image) in BCE loss')
parser.add_argument('--seg_loss_mode', type=str, default='BCE',  choices=['all' ,'softmax','BCE' , 'min'],
                    help='which loss to apply to heatmap') 
parser.add_argument('--use_w_seg_gt', type=str2bool, default=False, help='use w_seg_gt to adjust the weight of heatmap loss')
parser.add_argument('--lr', type=float, default=1e-3, metavar='N',
                    help='learning rate') 
## postprocessing for heatmap
parser.add_argument('--mask-inference-method', type=str, default='None', choices=['agf', 'sigmoid_per_predicted_class','bg_0', 'None'],
                    help='inference method for mask construction')  
parser.add_argument('--heatmap_scale', type=float, default=1, metavar='N',
                    help='scale factor applied to heatmap before BCE or softmax loss (after offset)')
parser.add_argument('--heatmap_offset', type=float, default=0, metavar='N',
                    help='offset applied to heatmap before BCE or softmax loss (before scaling)')
## visualizing prediction
parser.add_argument('--save_confidence_map', type=str2bool, default=False)

parser.add_argument('--no_skip_connection', action="store_true")
parser.add_argument('--fully_symmetric_unet', action="store_true")
parser.add_argument('--concate_before_block', action="store_true")
# v1 original 
# v2 --no_skip_connection should worse than original
# v3 --no_skip_connection --fully_symmetric_unet should equal to unorlled_lrp variant2 case
# V4 --concate_before_block may be better than original

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)
# torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

from models.baseline_multilabel import Baseline


os.makedirs("snapshot",exist_ok=True)


if __name__ == "__main__":
    # create the result folder
    try:
        os.stat(args.results_path)
    except :
        os.mkdir(args.results_path)

    # create the model (autoencoder)
    try:
        autoenc = Baseline(args)
        print("xyyyyyyyy")
        
    except KeyError:
        print('---------------------------------------------------------')
        print(f'Model architecture {args.model}-backbone: {args.backbone} not supported. ', end='')
        print('Maybe you can implement it?')
        print('---------------------------------------------------------')

    try:
        start=datetime.now()

        # print the args
        for keys,values in vars(args).items():
            print(f"{keys:30}: {values}")

        # initialize wandb
        if args.wandb != 'None':
            wandb.init(
                project=args.wandb,
                # config=args
            )
            #wandb.watch(ae.model,log_freq=10)

        # load pretrain model
        if args.pretrain_weight_name is not None and args.pretrain_weight_name !="imagenet":
            autoenc.load_pretrain_model(args.pretrain_weight_name)
       
        if args.pretrain_weight_name not in ["imagenet", None]:
            cp_epoch=int(args.pretrain_weight_name.split("/")[-1].split("_")[1])
        else: 
            cp_epoch=0

        ##test_interval
        # if args.loss_impact_bottleneck==0:
        #     test_interval_list={'20':60,"100":30,"500":10}
        #     test_interval=test_interval_list[str(args.n_masks)] if str(args.n_masks) in test_interval_list.keys() else 1
        # else:
        #     test_interval=10000


        # training part
        args.total_epochs=cp_epoch+ args.epochs
        print("Start offical training phase-------------------------------")
        for epoch in range(cp_epoch+1, cp_epoch+args.epochs + 1):
            autoenc.train(epoch)
            print(f'Epoch(train): [{epoch}/{cp_epoch+args.epochs}]  Time: {datetime.now()-start}')
            # if epoch%test_interval == 0:
            #     autoenc.test(epoch)
            #     print(f'Epoch(test): [{epoch}/{cp_epoch+args.epochs}]  Time: {datetime.now()-start}')

        end=datetime.now()
        print("Finish training-------------------------------")
        # print(f"Time for training {args.pre_epochs+args.epochs} is {end-start}.")
       

    except (KeyboardInterrupt, SystemExit):
        print("Manual Interruption")