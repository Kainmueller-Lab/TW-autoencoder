import argparse, os, sys
import numpy as np
import torch
from datetime import datetime
import wandb
import random
import logging


logger = logging.getLogger(__name__)
parser = argparse.ArgumentParser(
        description='Main function to call training for different AutoEncoders')
# dataset
parser.add_argument('--dataset', type=str, default='PASCAL',choices=['PASCAL']) 
parser.add_argument('--data_path',type=str,default="/home/xiaoyan/Documents/Data/",metavar='N')
parser.add_argument('--num_classes', type=int, default=21, metavar='N')
parser.add_argument('--seed', type=int, default=42, metavar='S') 
parser.add_argument('--crop_size', default=256, type=int)
parser.add_argument('--semisup_dataset', action="store_true", 
                    help='if set True, also include the image-level labelled data.')   
parser.add_argument('--num_labels', type=int, default=0, metavar='N',
                    help='number of labelled training data, set 0 to use all training data')
parser.add_argument('--uniform_masks', type=float, default=0.0, metavar='N',
                    help='switches the distribution of sampled masks to uniform distribution regarding the class labels')     

# model
parser.add_argument('--model', type=str, default='mt_unet', choices=['mt_unet','std_unet','unrolled_lrp'])       
parser.add_argument('--encoder', type=str, default='vgg16',choices=['vgg16','vgg16_bn','resnet18','resnet34','resnet50','resnet101'],
                    help='the backbone for the encoder')  
parser.add_argument('--no_cuda', action='store_true', default=False,
                    help='enables CUDA training')

# decoder xai method, if model == 'unrolled_lrp'
parser.add_argument('--xai', type=str, default='LRP_epsilon', choices=['LRP_epsilon','LRP_alphabeta','cLRP_type1','cLRP_type2'],
                    help='Where to store images')
parser.add_argument('--epsilon', type=float, default=1e-8, metavar='N',
                    help='epsilon value for LRP-epsilon rule') # related to xai args
parser.add_argument('--alpha', type=float, default=1, metavar='N',
                    help='alpha value for LRP-alpha beta rule') # related to xai args, default alpha=1(alpha>=1), beta=0

# pretrain
parser.add_argument('--pretrain_weight_name', type=str, default='None', metavar='N',
                    help='the path to save the pretrain weights')   # example : './snapshot/vgg_12_pre_train'
parser.add_argument('--pre_batch_size', type=int, default=0, metavar='N',
                    help='input batch size for training (default: 0)')
parser.add_argument('--pre_epochs', type=int, default=0, metavar='N',
                    help='number of pre-training epochs which only do classification')


# train
parser.add_argument('--batch_size', type=int, default=1, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=100, metavar='N',
                    help='number of epochs to train, including pretraining (default: 100)')
parser.add_argument('--results_path', type=str, default='results/', metavar='N', help='Where to store images')
parser.add_argument('--wandb', type=str, default='None', metavar='N', help='Sets up a wandb project')
parser.add_argument('--log_imgs', action="store_true",help='if log imgs to wandb during training')
parser.add_argument('--save_folder', type=str, default='test/', help='examples:"snapshot/", "/fast/AG_Kainmueller/xyu/"') #save_folder should be "snapshot/" or "/fast/AG_Kainmueller/xyu/"

# arguments for computational performance, if model == 'unrolled_lrp'
parser.add_argument('--iterative_gradients', action="store_true",
                    help='choice between computing the gradients iteratively for each class-related heatmap (memory efficient but more backwards passes) / or in one pass')

## loss desgin
parser.add_argument('--add_classification', action="store_true")
parser.add_argument('--loss_impact_seg', type=float, default=1, metavar='N',
                    help='scaling / impact of the segmentation loss')
parser.add_argument('--loss_impact_bottleneck', type=float, default=1, metavar='N',
                    help='scaling / impact of the bottleneck loss')            
parser.add_argument('--lr', type=float, default=1e-5, metavar='N', help='learning rate') 


args = parser.parse_args()


args.cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

logger_folder=args.save_folder if args.save_folder!='None' else 'test/'
os.makedirs(logger_folder, exist_ok=True)
logging.basicConfig(
    level=20,
    handlers=[
        logging.FileHandler(os.path.join(logger_folder,"run.log"), mode='a'),
        logging.StreamHandler(sys.stdout)
    ])
logger.info("%s", args)



if __name__ == "__main__":
    os.makedirs(args.results_path, exist_ok=True)
    os.makedirs("snapshot",exist_ok=True)

    # create the model (autoencoder)
    
    if args.model in ['mt_unet','std_unet']:
        from models.baseline_multilabel import Baseline
        autoenc=Baseline(args,logger)
    elif args.model == 'unrolled_lrp':
        from models.multilabel import TW_Autoencoder
        autoenc=TW_Autoencoder(args,logger)
    else:
        raise NotImplementedError
    

        
    # print the args
    for keys,values in vars(args).items():
        print(f"{keys:30}: {values}")
        
    # initialize wandb
    if args.wandb != 'None':
        wandb.init(project=args.wandb)
   
    try:
        start=datetime.now()

        # load pretrain model
        if args.pretrain_weight_name != 'None':
            autoenc.load_pretrain_model(args.pretrain_weight_name)
        
           
        # pretraining part
        if args.pre_epochs>0 and args.pre_batch_size>0:
            for epoch in range(1, args.pre_epochs + 1):
                autoenc.pretraining_train(epoch)
                autoenc.pretraining_test(epoch)
               
               
        ## offical training
        print("Start offical training phase-------------------------------")
        for epoch in range(args.pre_epochs + 1, args.epochs + 1):  
            autoenc.train(epoch)
            

        end=datetime.now()
        print("Finish training-------------------------------")
        print(f"Time for training {args.epochs} is {end-start}.")
       

    except (KeyboardInterrupt, SystemExit):
        print("Manual Interruption")

    
