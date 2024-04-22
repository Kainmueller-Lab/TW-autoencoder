import argparse, os, sys
import numpy as np
import torch
from datetime import datetime
import wandb
import random
  

parser = argparse.ArgumentParser(
        description='Main function to call training for different AutoEncoders')
# dataset
parser.add_argument('--dataset', type=str, default='PASCAL',choices=['PASCAL']) 
parser.add_argument('--data_path',type=str,default="/home/xiaoyan/Documents/Data/",metavar='N')
parser.add_argument('--num-classes', type=int, default=21, metavar='N')
parser.add_argument('--seed', type=int, default=42, metavar='S') 
parser.add_argument('--semisup_dataset', action="store_true", 
                    help='if set True, also include the image-level labelled data.')   
parser.add_argument('--num_labels', type=int, default=0, metavar='N',
                    help='number of labelled training data, set 0 to use all training data')
parser.add_argument('--uniform-masks', type=float, default=0.0, metavar='N',
                    help='switches the distribution of sampled masks to uniform distribution regarding the class labels')     

       
# model
parser.add_argument('--model', type=str, default='mt_unet', choices=['mt_unet','std_unet'])                   
parser.add_argument('--backbone', type=str, default='vgg16',choices=['vgg16','vgg16_bn','resnet18','resnet34','resnet50','resnet101'],
                    help='the backbone for the encoder')  #detail see documents baseline_multilabel
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')

# pretrain
parser.add_argument('--pretrain-weight-name', type=str, default='None', metavar='N',
                    help='the path to save the pretrain weights')   # example : './snapshot/vgg_12_pre_train'
parser.add_argument('--pre-batch-size', type=int, default=None, metavar='N',
                    help='input batch size for training (default: 32)')
parser.add_argument('--pre-epochs', type=int, default=10, metavar='N',
                    help='number of pre-training epochs which only do classification')


# train
parser.add_argument('--batch-size', type=int, default=40, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=100, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--results_path', type=str, default='results/', metavar='N', help='Where to store images')
parser.add_argument('--wandb', type=str, default='None', metavar='N', help='Sets up a wandb project')
# if log imgs to wandb during training
parser.add_argument('--log_imgs', action="store_true")
parser.add_argument('--save_folder', type=str, default=None, help='examples:"snapshot/", "/fast/AG_Kainmueller/xyu/"') #save_folder should be "snapshot/" or "/fast/AG_Kainmueller/xyu/"
## arguments for computational performance
parser.add_argument('--iterative-gradients', action="store_true",
                    help='choice between computing the gradients iteratively for each class-related heatmap (memory efficient but more backwards passes) / or in one pass')

## loss desgin
parser.add_argument('--add_classification', action="store_true")
parser.add_argument('--loss_impact_seg', type=float, default=1, metavar='N',
                    help='scaling / impact of the segmentation loss')
parser.add_argument('--loss_impact_bottleneck', type=float, default=1, metavar='N',
                    help='scaling / impact of the bottleneck loss')            
parser.add_argument('--lr', type=float, default=1e-3, metavar='N', help='learning rate') 

# arguments design for mt_unet, std_unet
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

        # training part
        args.total_epochs=cp_epoch+ args.epochs
        print("Start offical training phase-------------------------------")
        for epoch in range(cp_epoch+1, cp_epoch+args.epochs + 1):
            autoenc.train(epoch)
            print(f'Epoch(train): [{epoch}/{cp_epoch+args.epochs}]  Time: {datetime.now()-start}')
     

        end=datetime.now()
        print("Finish training-------------------------------")
        # print(f"Time for training {args.pre_epochs+args.epochs} is {end-start}.")
       

    except (KeyboardInterrupt, SystemExit):
        print("Manual Interruption")