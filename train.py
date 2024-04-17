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
            help='probability of sampling a mask') # outdated
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
parser.add_argument('--model', type=str, default='AE', metavar='N',
                    help='Which architecture to use')
parser.add_argument('--backbone', type=str, default='vgg16',choices=['vgg16','vgg16_bn','resnet18','resnet34','resnet50','resnet101','CNN_MNIST', 'efficientnet'],
                    help='the backbone for the encoder')  
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')

## decoder xai method
parser.add_argument('--xai', type=str, default='LRP_epsilon', choices=['LRP_epsilon','LRP_alphabeta','AGF', 'RAP','cLRP_type1','cLRP_type2'],
                    help='Where to store images')
parser.add_argument('--use-old-xai', type=str2bool, default=False, metavar='N',
                    help='if use old xai code to do sanity check for AGF and RAP') 
parser.add_argument('--epsilon', type=float, default=1e-8, metavar='N',
                    help='epsilon value for LRP-epsilon rule') # related to xai args
parser.add_argument('--alpha', type=float, default=1, metavar='N',
                    help='alpha value for LRP-alpha beta rule') # related to xai args, default alpha=1(alpha>=1), beta=0
parser.add_argument('--memory-efficient', type=str2bool, default=False, metavar='N',
                    help='if use memory efficient mode')  #arguments for computational performance
parser.add_argument('--detach_bias', type=str2bool, default=False)

# pretrain
parser.add_argument('--pretrain-weight-name', type=str_or_none, default='./snapshot/vgg_12_pre_train', metavar='N',
                    help='the path to save the pretrain weights')   # example : './snapshot/vgg_12_pre_train'
parser.add_argument('--pre-batch-size', type=int, default=32, metavar='N',
                    help='input batch size for training (default: 32)')
parser.add_argument('--pre-epochs', type=int, default=10, metavar='N',
                    help='number of pre-training epochs which only do classification')
parser.add_argument('--save-interval', type=int_or_none, default=None, metavar='N') #save_interval both exist in pretrain
parser.add_argument('--save_all_checkpoints', type=str2bool, default=False) # for heatmaps evolution figures
parser.add_argument('--test-interval', type=int, default=1, metavar='N',
                    help='test every n epochs') #save_interval both exist in train and pretrain
parser.add_argument('--use-earlystopping',type=str2bool,default=True,metavar='N', help='use earlier stopping')
parser.add_argument('--pretrain-weight-decay',type=float,default=0,metavar='N',
                    help='strength of l2 regularization with respect to the pretrained model (0 -> turned off)')

# train
parser.add_argument('--batch-size', type=int, default=1, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=100, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--results_path', type=str, default='results/', metavar='N', help='Where to store images')
parser.add_argument('--wandb', type=str, default='None', metavar='N', help='Sets up a wandb project')
# if log imgs to wandb during training
parser.add_argument('--log_imgs', type=str2bool, default=True)
parser.add_argument('--log_tables', type=str2bool, default=False)
parser.add_argument('--clip',type=float,default=0,metavar='N', help='gradient clip (0 -> turned off)')
parser.add_argument('--save_folder', type=str, default=None, help='examples:"snapshot/", "/fast/AG_Kainmueller/xyu/"') #save_folder should be "snapshot/" or "/fast/AG_Kainmueller/xyu/"

# arguments for computational performance
parser.add_argument('--iterative-gradients', type=str2bool, default=False, metavar='S',
                    help='choice between computing the gradients iteratively for each class-related heatmap (memory efficient but more backwards passes) / or in one pass')

# loss desgin
parser.add_argument('--loss_impact_seg', type=float, default=0.1, metavar='N',
                    help='scaling / impact of the segmentation loss')
parser.add_argument('--loss_impact_bottleneck', type=float, default=1, metavar='N',
                    help='scaling / impact of the bottleneck loss')            
parser.add_argument('--bg_weight_in_BCE_hm_loss', type=float, default=0, metavar='N',
                    help='weight for class 0 in BCE loss')
parser.add_argument('--nonexist_weight_in_BCE_hm_loss', type=float, default=0.1, metavar='N',
                    help='weight for non-existing classes (per image) in BCE loss')
parser.add_argument('--seg_loss_mode', type=str, default='softmax', choices=['all' ,'softmax','BCE' , 'min','valley'],
                    help='which loss to apply to heatmap') 
parser.add_argument('--use_w_seg_gt', type=str2bool, default=False, help='use w_seg_gt to adjust the weight of heatmap loss')
parser.add_argument('--lr', type=float, default=1e-4, metavar='N', help='learning rate')
# postprocessing for heatmap
parser.add_argument('--mask-inference-method', type=str, default='None', choices=['agf', 'sigmoid_per_predicted_class','bg_0', 'None','valley'],
                    help='inference method for mask construction')  
parser.add_argument('--heatmap_scale', type=float, default=1, metavar='N',
                    help='scale factor applied to heatmap before BCE or softmax loss (after offset)')
parser.add_argument('--heatmap_offset', type=float, default=0, metavar='N',
                    help='offset applied to heatmap before BCE or softmax loss (before scaling)')

# visualizing prediction
parser.add_argument('--save_confidence_map', type=str2bool, default=False)

# rest (some arguments for the multiclass.py)
parser.add_argument('--embedding-size', type=int, default=32, metavar='N',
                    help='how many batches to wait before logging training status')

# ablation test
parser.add_argument('--normal_relu', action="store_true") # variant1 normal_relu==True
parser.add_argument('--normal_deconv', action="store_true") # variant2 normal_relu==True and normal_deconv==True
parser.add_argument('--normal_unpool', action="store_true") # variant3 normal_unpool=True normal_relu==True and normal_deconv==True
parser.add_argument('--multiply_input', action="store_true") 
parser.add_argument('--remove_heaviside', action="store_true") 
parser.add_argument('--remove_last_relu', action="store_true") 
parser.add_argument('--add_bottle_conv', action="store_true") 

# verify experiments
parser.add_argument('--only_send_labeled_data', action="store_true") # only pass through labeled data

# variant1_1: --normal_relu --remove_last_relu --multiply_input 
# variant1_2: --normal_relu --remove_last_relu --remove_heaviside (--multiply_input) 
# variant2_1_1: --normal_relu --normal_deconv --remove_heaviside --remove_last_relu
# variant2_1_2: --normal_relu --normal_deconv --remove_heaviside --remove_last_relu --only_send_labeled_data
# variant2_2: --normal_relu --normal_deconv --remove_heaviside --remove_last_relu --add_bottle_conv 
# variant3: --normal_relu --normal_deconv --normal_unpool --remove_heaviside --remove_last_relu

args = parser.parse_args()

# ablation test
# if normal_deconv is set to True, the normal_relu is set to True automatically
# if args.normal_deconv and not args.normal_unpool:
#     args.normal_relu=True
#     args.iterative_gradients=False
#     # args.remove_heaviside=False
#     print("The argument normal_relu=True and iterative_gradients=False automatically")
#     print("Unrolled_lrp model's ablation test2")

if args.normal_unpool:
    args.normal_relu=True
    args.normal_deconv=True
    args.iterative_gradients=False
    # args.remove_heaviside=False
    print("The argument normal_relu and args.normal_deconv=True and iterative_gradients=False automatically")
    print("Unrolled_lrp model's ablation test3")

if args.normal_relu== True and args.normal_deconv==False:
    print("Unrolled_lrp model's ablation test1")

if (args.normal_relu== False and args.normal_deconv==False) or (args.normal_deconv==True):
    args.multiply_input = False

args.cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

if args.dataset in ['MNIST','EMNIST','FashionMNIST']:
    NotImplementedError(" the models.multiclass may need to be modified")
elif args.dataset in ['Lizard','PASCAL']:
    from models.multilabel import AE
else:
    NotImplementedError("Dataset not supported")

ae = AE(args)
architectures = {'AE':  ae}
os.makedirs("snapshot",exist_ok=True)


if __name__ == "__main__":
    # create the result folder
    try:
        os.stat(args.results_path)
    except :
        os.mkdir(args.results_path)

    # create the model (autoencoder)
    try:
        autoenc = architectures[args.model]
    except KeyError:
        print('---------------------------------------------------------')
        print('Model architecture not supported. ', end='')
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
        if args.pretrain_weight_name is not None:
            pretrained_model_name = args.pretrain_weight_name.split("/")[-1]
            cp_epoch=int(pretrained_model_name.split("_")[1])
            autoenc.load_pretrain_model(args.pretrain_weight_name,cp_epoch)
        else:
            cp_epoch=0
           
        # training part
        ## if checkpoint epoch < pretrain epochs, pretrain-->official train
        if cp_epoch<args.pre_epochs:
            ## pre training
            print("Start pre training phase-------------------------------")
            for epoch in range(cp_epoch+1, args.pre_epochs + 1):
                start_epoch=epoch
                autoenc.pretraining_train(epoch)
                if epoch%args.test_interval == 0:
                    autoenc.pretraining_test(epoch)  
                if args.use_earlystopping and autoenc.early_stopping.early_stop==True:
                    print(f'Early stopping after epoch {epoch}')
                    autoenc.save_model(epoch,"pre_train")
                    break              
               

            ## offical training
            print("Start offical training phase-------------------------------")
            args.total_epochs=start_epoch+ args.epochs
            for epoch in range(start_epoch+1,start_epoch+ args.epochs + 1):  
                autoenc.train(epoch)
                if epoch%args.test_interval == 0:
                    autoenc.test(epoch)

        ## if checkpoint epoch > pretrain epochs, -->official train
        else:
            ## offical training
            args.total_epochs=cp_epoch+args.epochs
            print("Start offical training phase-------------------------------")
            for epoch in range(cp_epoch+1, cp_epoch+args.epochs + 1):
                autoenc.train(epoch)
                # if epoch%args.test_interval == 0:
                #     autoenc.test(epoch)

        end=datetime.now()
        print("Finish training-------------------------------")
        print(f"Time for training {args.pre_epochs+args.epochs} is {end-start}.")
       

    except (KeyboardInterrupt, SystemExit):
        print("Manual Interruption")

    
