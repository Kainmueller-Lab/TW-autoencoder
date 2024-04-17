import argparse, os, sys
from socket import SOCK_SEQPACKET
from genericpath import exists
import numpy as np
import torch
from datetime import datetime
import wandb
import random
sys.path.append("..")  # Adds the parent directory to the sys.path
from baseline_arch.multi_task_unet import MTUNet
from models.multilabel import Network
from datasets import PASCAL_dataset
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
import matplotlib.pyplot as plt

from skimage import color



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


def draw_colors(mask, cmap):
    '''
    mask: H x W numpy array containg class index per pxl
    colors: array with colors for every class
    '''
    img = np.zeros(mask.shape+(3,)) # H x W x 3
    classes = np.unique(mask)
    for c in classes:
        class_pxls = (mask == c)
        # img[class_pxls] = to_rgb(colors[c])
        img[class_pxls]=np.array(cmap[c])
    
    return img.astype(np.uint8)


def color_map(N=256, normalized=False):
    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)

    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7-j)
            g = g | (bitget(c, 1) << 7-j)
            b = b | (bitget(c, 2) << 7-j)
            c = c >> 3

        cmap[i] = np.array([r, g, b])

    cmap = cmap/255 if normalized else cmap
    return cmap

def normalize_image_percentile(in_img):
    img_max=np.percentile(in_img,99)
    img_min=np.percentile(in_img,0)
    # range_im=img_max-img_min
    # if range_im>0:
    #     norm_img=(in_img-img_min)/range_im
    # else:
    #     norm_img=in_img-img_min
    norm_img=in_img/img_max
    norm_img=norm_img.clip(0.0,1.0) 
    return norm_img

def color_heatmap(img):
    
    hue = np.ones_like(img)
    hue[img>0]=1.0 # red hue value
    hue[img<0]=0.6667 # blue hue value
    hue[img==0]=0.3333 # this is not really needed
    # sat = normalize_image_percentile(np.abs(img))
    # sat = np.abs(img)
    sat= np.ones_like(img)

    # brightness = np.ones_like(img) # add saturation, otherwise very dark pixels do not get colored
    brightness=np.abs(img)

    blended_hsv = np.stack([hue,sat, brightness], axis=2)
    blended_hsv_rgb = np.round(color.hsv2rgb(blended_hsv)*255).astype(np.uint8)/255
    return blended_hsv_rgb 

def color_map(N=256, normalized=False):
    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)

    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7-j)
            g = g | (bitget(c, 1) << 7-j)
            b = b | (bitget(c, 2) << 7-j)
            c = c >> 3

        cmap[i] = np.array([r, g, b])

    cmap = cmap/255 if normalized else cmap
    return cmap


def draw_colors(mask, cmap):
    '''
    mask: H x W numpy array containg class index per pxl
    colors: array with colors for every class
    '''
    img = np.zeros(mask.shape+(3,)) # H x W x 3
    classes = np.unique(mask)
    for c in classes:
        class_pxls = (mask == c)
        # img[class_pxls] = to_rgb(colors[c])
        img[class_pxls]=np.array(cmap[c])
    
    return img.astype(np.uint8)


def load_checkpoint(args, model,name, iteration, n_masks):
    # print("-----------load checkpoint weight-----------------")
    checkpoint_name=f"{args.checkpoint_folder}"+f"{args.backbone}_seed{args.seed}_pre_epochs{args.pre_epochs}" + \
        f"/{name}_{args.backbone}_iter{iteration}_lab{n_masks}_lr{args.lr}_bs{args.batch_size}.pth" 
    # print(f"Loading the following checkpoint: {checkpoint_name}")
    checkpoint=torch.load(checkpoint_name)
    model.load_state_dict(checkpoint)
    return model



def heatmap_plot(args,model, test_loader, dict,device, iteration, n_masks, save_folder):

    print(f"--------------------------start for case: label-{n_masks} iter-{iteration}---------------")
    for batch_idx, (data, class_labels, sem_gts, sem_gt_exists) in tqdm(enumerate(test_loader)):
        sem_gts = sem_gts.long().to(device)
        data = data.to(device)
        class_labels = class_labels.to(device)

        if batch_idx in img_idx_list:
            # main part for heatmap calculation  
            # predict heatmap
            if dict['name']=="mt_unet":
                heatmaps,_ = model(data)
            else:
                _ ,heatmaps= model(data)

            # FIX THE BATCHSIZE=1
            class_labels=class_labels[0].detach().cpu() # C
            class_labels[0]=0 #shut down the background class


            #############
            if args.all_classes==False:
                exist_c_idx=np.nonzero(class_labels.numpy())[0] # picke the non zero element index along dim=0 #C 
            else:
                exist_c_idx=np.arange(0,21)


           
            if len(exist_c_idx)>2:
                pass
            else:

                # pick the class-wise heatmaps along dim=0
                if n_masks==0:
                    if args.no_softmax:
                        c_heatmaps=heatmaps.detach().cpu()[0,exist_c_idx] # num_exist_classes x H x W
                        if args.normalize_image_percentile:
                            c_heatmaps=[normalize_image_percentile(item)/2 for item in c_heatmaps]
                            c_heatmaps=torch.stack(c_heatmaps,dim=0)
                        else:
                            c_heatmaps=[item/(2*torch.abs(item).max())for item in c_heatmaps] # num_exist_classes x H x W, fix range in (-1,1)
                            for item in c_heatmaps:
                                print(f"item max and min {item.max():.2f} and {item.min():.2f} after normalize, should be in range(-0.5,0.5). The zero relevance value is always centered.")
                            c_heatmaps=torch.stack(c_heatmaps,dim=0)
                    else:

                        c_heatmaps=heatmaps.detach().cpu()[0,exist_c_idx] # num_exist_classes x H x W
                        max_value,_=torch.max(torch.abs(c_heatmaps), dim=1, keepdim=True)
                        max_value,_=torch.max(max_value, dim=2, keepdim=True) # num_exist_classes x H x W, make sure the max value is
                        c_heatmaps=c_heatmaps/max_value
                        if args.normalize_image_percentile:
                            c_heatmaps=[normalize_image_percentile(item) for item in c_heatmaps]
                            c_heatmaps=torch.stack(c_heatmaps,dim=0)

                else:       
                    if args.no_softmax: 
                        c_heatmaps=heatmaps.detach().cpu()[0,exist_c_idx] # num_exist_classes x H x W
                        if args.normalize_image_percentile:
                            c_heatmaps=[normalize_image_percentile(item) for item in c_heatmaps]
                            c_heatmaps=torch.stack(c_heatmaps,dim=0)
                        else:
                            c_heatmaps=[item/(2*torch.abs(item).max())for item in c_heatmaps] # num_exist_classes x H x W, fix range in (-1,1)
                            for item in c_heatmaps:
                                print(f"item max and min {item.max():.2f} and {item.min():.2f} after normalize, should be in range(-0.5,0.5). The zero relevance value is always centered.")
                            c_heatmaps=torch.stack(c_heatmaps,dim=0)
                    else:
                        c_heatmaps=torch.softmax(heatmaps.detach().cpu(),dim=1)[0,exist_c_idx] # num_exist_classes x H x W


                assert c_heatmaps.shape[0]==len(exist_c_idx),f"the shape of c_heatmaps {c_heatmaps.shape} is  \
                            not match with num of exists classess {len(exist_c_idx)}"
                for c_heatmap, class_idx in zip(c_heatmaps,exist_c_idx):
                    # c_heatmap H x W

                    # color and save heatmap
                    if args.jet_color:

                        alpha = 1
                        a=plt.cm.jet(c_heatmap.numpy())[:, :, :3] * 255 # HxWx4->HxWx3
                        b=data[0].detach().permute(1,2,0).cpu().numpy()* 255 #3xHxW-> HxWx3
                        img_feature=(a* alpha + b * (1 - alpha)).astype(np.uint8)


                        img_folder=save_folder+f"/jet_color_lab{n_masks}_softmax_{args.softmax}/"
                        img_name=f"{batch_idx}_iter_{iteration}_class_{class_dict[class_idx]}.png"
                        os.makedirs(img_folder,exist_ok=True)

                        im=Image.fromarray(img_feature)
                        im.save( img_folder+ img_name)

                    else:

                        alpha = 1

                        if args.use_pil_library:

                            c_heatmap=c_heatmap-0.5

                            a=(color_heatmap(c_heatmap)*255).astype(np.uint8)  
                            b=(data[0].detach().permute(1,2,0).cpu().numpy()* 255).astype(np.uint8)


                            img_folder=save_folder+f"/red_blue_color_lab{n_masks}_softmax_{args.softmax}/"
                            img_name=f"{batch_idx}_iter_{iteration}_class_{class_dict[class_idx]}.png"
                            os.makedirs(img_folder,exist_ok=True)

                            a=Image.fromarray(a).convert("RGBA")
                            b=Image.fromarray(b).convert("RGBA")
                            im=Image.blend(a,b, alpha)
                            im.save( img_folder+ img_name)

                        else:
                            if args.no_softmax: 
                                c_heatmap=c_heatmap+0.5

                         

                            a=plt.cm.seismic(c_heatmap.numpy())[:, :, :3] * 255 # use plt.cm.seismic 0.5=white                               
                            b=data[0].detach().permute(1,2,0).cpu().numpy()* 255
                            img_feature=(a* alpha + b * (1 - alpha)).astype(np.uint8)


                            img_folder=save_folder+f"/red_blue_color_lab{n_masks}_softmax_{args.softmax}/"
                            img_name=f"{batch_idx}_iter_{iteration}_class_{class_dict[class_idx]}.png"
                            os.makedirs(img_folder,exist_ok=True)

                            im=Image.fromarray(img_feature)
                            im.save( img_folder+ img_name)



                # also save the original data
                # draw image
                # if n_masks==0:
                #     img_folder=save_folder + f"/raw/"
                #     os.makedirs(img_folder,exist_ok=True)
                #     img_data = np.uint8(data[0].permute(1,2,0).cpu().numpy()*255)
                #     img_name=f"{batch_idx}_raw.png"
                #     im=Image.fromarray(img_data)
                #     im.save( img_folder+ img_name)

                #     img_folder=save_folder + f"/sem_gt/"
                #     os.makedirs(img_folder,exist_ok=True)
                #     img_gt = draw_colors(sem_gts[0].cpu().numpy(),color_map())
                #     img_name=f"{batch_idx}_sem_gt.png"
                #     im=Image.fromarray(img_gt)
                #     im.save( img_folder+ img_name)
            
        elif batch_idx> max(img_idx_list):
            break
        else:
            pass
    return 


def paste_img(path, axs, i, j,num_rows):
    img=np.array(Image.open(path))
    if num_rows==1:
        axs[j].imshow(img)
        axs[j].axis('off') 
    else:
        axs[i,j].imshow(img)
        axs[i,j].axis('off') 

            
            
    # set colorscheme for 20 classes TODO make disjoint appearing colors for co-present classes

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
    parser.add_argument('--n-masks', type=int_or_none, default=None, metavar='N', help='number of masks applied during training of the original training masks')
    parser.add_argument('--uniform-masks', type=float, default=0.0, metavar='N', help='switches the distribution of sampled masks to uniform distribution regarding the class labels')
    parser.add_argument('--trafo-mode', type=int, default=0, metavar='N', help='different types of augmentations')

    # model
    parser.add_argument('--name', type=str, default="std_unet",choices=['std_unet','mt_unet','lrp0'])
    parser.add_argument('--backbone', type=str, default='resnet50',choices=['vgg16','vgg16_bn','resnet18','resnet34','resnet50','resnet101'], help='the backbone for the encoder')
    parser.add_argument('--iteration', type=int, default=0,help="related to loaded checkpoint name")
    parser.add_argument('--pre_epochs', type=int, default=10,help="related to loaded checkpoint name")
    parser.add_argument('--no-cuda', action='store_true', default=False, help='enables CUDA training')

    # train
    parser.add_argument('--batch-size', type=int, default=10, help="related to loaded checkpoint name")
    parser.add_argument('--checkpoint_folder', type=str, default='../snapshot/all_checkpoints/') 

    # loss desgin
    parser.add_argument('--lr', type=float, default=1e-5, metavar='N', help='learning rate')
    parser.add_argument('--results_path', type=str, default='results/', metavar='N', help='Where to store images')
    parser.add_argument('--cuda-dev', type=str, default='cuda', metavar='N')
    
    # plot
    parser.add_argument('--jet_color', type=str2bool, default=False)
    parser.add_argument('--use_pil_library', type=str2bool, default=False)

    parser.add_argument('--all_classes', type=str2bool, default=False)
    parser.add_argument('--normalize_image_percentile', type=str2bool, default=False)
    parser.add_argument('--function', type=str, default='generate',choices=['generate','stitch'])
    parser.add_argument('--no_softmax', type=str2bool, default=False)
    
    
    

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device(args.cuda_dev if args.cuda else "cpu")
 
    args.softmax=True if args.no_softmax==False else False
    
    
    ## load test dataset according to args
    print("-----------load test dataset-----------------")
    random.seed(100)
    prefixed_args={"path": args.data_path, 
                    "seed": args.seed,
                    "n_masks": args.n_masks,
                    "uniform_masks": args.uniform_masks,
                    "reduce_classes":args.reduce_classes,
                    "weighted_sampling": args.weighted_mask_sampling,
                    "normalize": args.normalize,
                    "trafo_mode": args.trafo_mode
                    }


  
    ## manual input information for the checkpoints

   

    end=279000
    lrp0_dict={
        'name':'lrp0',
        'list_iter':[item for item in range(159000, end+1,3000)], #change TODO
        'list_n_masks':[20,100,500,None],
        'test_batch_size':1,
        'add_xlabel':True
    }
 

    # lrp0_dict['list_iter']=[159000,162000,177000,231000,237000]
    # img_idx_list=[1035]
    # img_idx_list=(np.arange(0,1449)).tolist()

    lrp0_dict['list_n_masks']=[100,500, None]
    lrp0_dict['list_iter']=[item for item in range(159000, end+1,3000)]
    # img_idx_list=[4,9,20,24,36,144,159,83,108,143,190,213]
    # lrp0_dict['list_iter']=[165000,168000,171000,180000,195000,252000]
 
    img_idx_list=[5,9,138,623,745,801,809]
    


    # img_dict={
    #     'img_idx':617,

    #     '20': [159000,162000,174000,183000,192000,204000],
    #     '100':[159000,162000,168000,174000,192000,234000],
    #     '500':[159000,168000,177000,195000,237000,273000],
    #     'None':[159000,162000,177000,204000,207000,249000]
    # }
    # # lrp0_dict['list_n_masks']=[100, 500]
    # img_dict={
    #     'img_idx':809,
    #     'list_n_masks':[100,500],

    #     # '20': [159000,162000,165000,171000,174000,180000],
    #     # '100':[159000,162000,168000,189000,204000,228000],
    #     # '500':[159000,165000,189000,207000,234000,252000],
    #     '100':[162000,168000,174000,189000,204000,228000],
    #     '500':[165000,189000,207000,234000,249000,252000],
    # #     'None':[159000,165000,186000,198000,216000,249000]
    # }
    # img_dict={
    #     'img_idx':463,

    #     '20': [159000,162000,165000,177000,204000,216000],
    #     '100':[159000,162000,165000,177000,204000,216000],
    #     '500':[159000,171000,198000,201000,216000,246000],
    #     'None':[159000,162000,186000,216000,249000,273000]
    # }


   
    big_dict=[

    #################################################lab100
         {
        'img_idx':5,
        'list_n_masks':[100],
        '100':[165000,168000,177000,186000,201000,249000]
    },
    {
        'img_idx':9,
        'list_n_masks':[100],
        '100':[162000,165000,168000,174000,189000,195000]
    },
    {
        'img_idx':138,
        'list_n_masks':[100],
        '100':[165000,171000,189000,207000,228000,270000]
    },
   {
        'img_idx':623,
        'list_n_masks':[100],
        '100':[162000,165000,168000,171000,180000,195000]
    },
   {
        'img_idx':745,
        'list_n_masks':[100],
        '100':[162000,165000,168000,183000,201000,237000]
    },
   {
        'img_idx':801,
        'list_n_masks':[100],
        '100':[165000,168000,171000,180000,195000,252000]
    },
    {
        'img_idx':809,
        'list_n_masks':[100],
        '100':[165000,174000,189000,216000,231000,252000]
    },

    #################################################lab500
        {
        'img_idx':5,
        'list_n_masks':[500],
        '500':[165000,168000,171000,177000,192000,219000]
    },
    {
        'img_idx':9,
        'list_n_masks':[500],
        '500':[165000,171000,195000,222000,243000,258000]
    },
    {
        'img_idx':138,
        'list_n_masks':[500],
        '500':[165000,183000,213000,231000,237000,246000]
    },

    {
        'img_idx':623,
        'list_n_masks':[500],
        '500':[162000,165000,174000,207000,240000,267000]
    },

    {
        'img_idx':745,
        'list_n_masks':[500],
        '500':[162000,168000,183000,210000,234000,246000]
    },

    {
        'img_idx':801,
        'list_n_masks':[500],
        '500':[162000,168000,183000,201000,234000,246000]
    },

    {
        'img_idx':809,
        'list_n_masks':[500],
        '500':[165000,174000,198000,222000,234000,246000]
    },

#################################################labNone
    {
        'img_idx':5,
        'list_n_masks':[None],
        'None':[162000,171000,186000,198000,261000,279000]
    },
    {
        'img_idx':9,
        'list_n_masks':[None],
        'None':[162000,171000,207000,237000,261000,279000]
    },

    {
        'img_idx':138,
        'list_n_masks':[None],
        'None':[162000,168000,189000,198000,258000,276000]
    },

    {
        'img_idx':623,
        'list_n_masks':[None],
        'None':[162000,168000,195000,234000,273000,276000]
    },
    {
        'img_idx':745,
        'list_n_masks':[None],
        'None':[165000,186000,210000,240000,267000,276000]
    },

    {
        'img_idx':801,
        'list_n_masks':[None],
        'None':[162000,174000,222000,246000,252000,276000]
    },
    {
        'img_idx':809,
        'list_n_masks':[None],
        'None':[162000,174000,186000,213000,228000,252000]
    }


    ]









    
    
    testing_dataset = PASCAL_dataset(**prefixed_args,mode="test",full_supervison=True)
    test_loader = torch.utils.data.DataLoader(testing_dataset, batch_size=1)
    class_dict=testing_dataset.new_classes


    save_folder=f"../results/heatmaps/{lrp0_dict['name']}_{args.backbone}_lr{args.lr}_bs{args.batch_size}"

    ###############generate_figures###############
    if args.function== 'generate':
        model=intialize_model(args, lrp0_dict)
        #load the pretrain checkpoint
        checkpoint_name=f"../snapshot/"+ f"{args.backbone}_{args.pre_epochs}_pre_train_21.pth"
   
        model.to(device)
        model.eval()
        with torch.no_grad():
            # checkpoint=torch.load(checkpoint_name)
            # model.load_state_dict(checkpoint)
            # heatmap_plot(args,model, test_loader, lrp0_dict,device, iteration=156760, n_masks=0, save_folder=save_folder )
    
            for  n_masks in lrp0_dict['list_n_masks']:
                for iteration in lrp0_dict['list_iter']:
                    model=load_checkpoint(args, model,lrp0_dict['name'], iteration, n_masks)
                    heatmap_plot(args,model, test_loader, lrp0_dict,device, iteration, n_masks,save_folder)
    else:
    ###############stitch the images together###############
        color_style="jet_color" if args.jet_color==True else "red_blue_color"
        for img_dict in big_dict:
            for n_masks in img_dict['list_n_masks']:
                for batch_idx, (data, class_labels, sem_gts, sem_gt_exists) in tqdm(enumerate(test_loader)):
                    # if batch_idx in img_idx_list:
                    if batch_idx == img_dict['img_idx']:
                        img_idx=batch_idx
                        # FIX THE BATCHSIZE=1
                        class_labels=class_labels[0] # C
                        class_labels[0]=0 #shut down the background class
                        
                        if args.all_classes==False:
                            exist_c_idx=np.nonzero(class_labels.numpy())[0] # picke the non zero element index along dim=0 #C 
                        else:
                            exist_c_idx=np.arange(0,21)
                    
                        fig, axs = plt.subplots(len(exist_c_idx), len(img_dict[f'{n_masks}'])+3, figsize=( 4*(3+len(img_dict[f'{n_masks}'])),4*(len(exist_c_idx))), sharey=True)  
                        for i,k in enumerate(exist_c_idx):
                            # for j in range(len(lrp0_dict['list_iter'])+3):  
                            for j in range(len(img_dict[f'{n_masks}'])+3):
                                if j==0:
                                    # first column is for the raw data
                                    fig_path="../results/heatmaps/"+ f"{lrp0_dict['name']}_{args.backbone}_lr{args.lr}_bs{args.batch_size}/" + \
                                                    f"raw/"
                                    path=fig_path+f"{batch_idx}_raw.png"
                                    paste_img(path, axs, i, j,len(exist_c_idx))    
                                elif j==1:
                                    # second column put the pretrain checkpoint figure
                                    iteration=156760
                                    fig_path="../results/heatmaps/"+ f"{lrp0_dict['name']}_{args.backbone}_lr{args.lr}_bs{args.batch_size}/" + \
                                                    f"{color_style}_lab0_softmax_False/"
                                    path=fig_path+f"{img_idx}_iter_{iteration}_class_{class_dict[k]}.png"
                                    paste_img(path, axs, i, j,len(exist_c_idx))

                                elif j>1 and j< len(img_dict[f'{n_masks}'])+2:
                                    # iteration=lrp0_dict['list_iter'][j-2]
                                    iteration=img_dict[f'{n_masks}'][j-2]
                                    print(f"j-2, iteration {j-2} {iteration}")


                                    fig_path="../results/heatmaps/"+ f"{lrp0_dict['name']}_{args.backbone}_lr{args.lr}_bs{args.batch_size}/" + \
                                            f"{color_style}_lab{n_masks}_softmax_{args.softmax}/"
                                    path=fig_path+f"{img_idx}_iter_{iteration}_class_{class_dict[k]}.png"
                                    paste_img(path, axs, i, j,len(exist_c_idx))
                                else:
                                    # last column put the GT
                                    fig_path="../results/heatmaps/"+ f"{lrp0_dict['name']}_{args.backbone}_lr{args.lr}_bs{args.batch_size}/" + \
                                            f"sem_gt/"
                                    path=fig_path+f"{img_idx}_sem_gt.png"
                                    paste_img(path, axs, i, j,len(exist_c_idx))

                            # add y label
                            if len(exist_c_idx)==1:
                                axs[i].text(-0.3,0.5, f"{class_dict[k]}", size=22, ha="center", 
                                            transform=axs[i].transAxes)
                            else:
                                axs[i,0].text(-0.3,0.5, f"{class_dict[k]}", size=22, ha="center", 
                                            transform=axs[i,0].transAxes)
                            
                        # add x label, can move out of the loop
                        if len(exist_c_idx)==1:
                            plt.subplots_adjust(wspace = 0.05)
                            # for j in range(len(lrp0_dict['list_iter'])+3):
                                
                            #     if j==0:
                            #         axs[j].text(0.5,-0.2, "Image", size=23, ha="center", 
                            #                     transform=axs[j].transAxes)
                            #     elif j==1:
                            #         # axs[j].text(0.5,-0.2, str(156760), size=23, ha="center", 
                            #         #             transform=axs[j].transAxes)
                            #         pass
                            #     elif j>1 and j< len(img_dict[f'{n_masks}'])+2:
                            #         # axs[j].text(0.5,-0.2, str(lrp0_dict['list_iter'][j-2]), size=23, ha="center", 
                            #         #             transform=axs[j].transAxes)
                            #         pass
                            #     else:
                            #         axs[j].text(0.5,-0.2, "GT", size=23, ha="center", 
                            #                     transform=axs[j].transAxes)
                        else:
                            plt.subplots_adjust(wspace = 0.05,hspace=0.05)
                            # last_row=len(exist_c_idx)-1
                            # for j in range(len(lrp0_dict['list_iter'])+3):
                                
                            #     if j==0:
                            #         axs[last_row,j].text(0.5,-0.2, "Image", size=23, ha="center", 
                            #                     transform=axs[last_row,j].transAxes)
                            #     elif j==1:
                            #         # axs[last_row,j].text(0.5,-0.2, str(156760), size=23, ha="center", 
                            #         #             transform=axs[last_row,j].transAxes)
                            #         pass
                            #     elif j>1 and j< len(lrp0_dict['list_iter'])+2:
                            #         # axs[last_row,j].text(0.5,-0.2, str(156760), size=23, ha="center", 
                            #         #             transform=axs[last_row,j].transAxes)
                            #         pass
                            #     else:
                            #         axs[last_row,j].text(0.5,-0.2, "GT", size=23, ha="center", 
                            #                     transform=axs[last_row,j].transAxes)


                                # if need to add the text
                        # plt.tight_layout()
                        path_save=f"../results/heatmaps/lrp0_{args.backbone}_lr{args.lr}_bs{args.batch_size}/final_version4/{color_style}_softmax_{args.softmax}/lab{n_masks}"
                        os.makedirs(path_save, exist_ok=True)
                        plt.savefig(path_save + '/' + f'overview_{img_idx}_lab{n_masks}_iter_156760-{end}'+'.png',bbox_inches='tight')
                        plt.close(fig)

                    else:
                        pass
