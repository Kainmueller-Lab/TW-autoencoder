import argparse, os, sys
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



def heatmap_plot(model, test_loader, dict,device, iteration, n_masks, save_folder):

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
            exist_c_idx=np.nonzero(class_labels.numpy())[0] # picke the non zero element index along dim=0 #C 


            # pick the class-wise heatmaps along dim=0
            if n_masks==0:
                c_heatmaps=heatmaps.detach().cpu()[0,exist_c_idx] # num_exist_classes x H x W
                c_heatmaps=c_heatmaps/c_heatmaps.max()
            else:
                c_heatmaps=torch.softmax(heatmaps.detach().cpu(),dim=1)[0,exist_c_idx] # num_exist_classes x H x W

           
            assert c_heatmaps.shape[0]==len(exist_c_idx),f"the shape of c_heatmaps {c_heatmaps.shape} is  \
                        not match with num of exists classess {len(exist_c_idx)}"
            for c_heatmap, class_idx in zip(c_heatmaps,exist_c_idx):
                # c_heatmap H x W
         
                # color and save heatmap
                if args.jet_color:

                    alpha = 0.5
                    a=plt.cm.jet(c_heatmap.numpy())[:, :, :3] * 255 # HxWx4->HxWx3
                    b=data[0].detach().permute(1,2,0).cpu().numpy()* 255 #3xHxW-> HxWx3
                    img_feature=(a* alpha + b * (1 - alpha)).astype(np.uint8)
               

                    img_folder=save_folder+f"/jet_color_lab{n_masks}/"
                    img_name=f"{batch_idx}_iter_{iteration}_class_{class_dict[class_idx]}.png"
                    os.makedirs(img_folder,exist_ok=True)

                    im=Image.fromarray(img_feature)
                    im.save( img_folder+ img_name)
                  
                else:

                    alpha = 0.5
                
                    if args.use_pil_library:

                        c_heatmap=c_heatmap-0.5
                        
                        a=(color_heatmap(c_heatmap)*255).astype(np.uint8)  
                        b=(data[0].detach().permute(1,2,0).cpu().numpy()* 255).astype(np.uint8)
        

                        img_folder=save_folder+f"/red_blue_color_lab{n_masks}/"
                        img_name=f"{batch_idx}_iter_{iteration}_class_{class_dict[class_idx]}.png"
                        os.makedirs(img_folder,exist_ok=True)

                        a=Image.fromarray(a).convert("RGBA")
                        b=Image.fromarray(b).convert("RGBA")
                        im=Image.blend(a,b, alpha)
                        im.save( img_folder+ img_name)

                    else:
                               
                        a=plt.cm.seismic(c_heatmap.numpy())[:, :, :3] * 255 # use plt.cm.seismic 0.5=white                               
                        b=data[0].detach().permute(1,2,0).cpu().numpy()* 255
                        img_feature=(a* alpha + b * (1 - alpha)).astype(np.uint8)


                        img_folder=save_folder+f"/red_blue_color_lab{n_masks}/"
                        img_name=f"{batch_idx}_iter_{iteration}_class_{class_dict[class_idx]}.png"
                        os.makedirs(img_folder,exist_ok=True)

                        im=Image.fromarray(img_feature)
                        im.save( img_folder+ img_name)
                    
                 

            # also save the original data
            # draw image
            # img_data = np.uint8(data[0].permute(1,2,0).cpu().numpy()*255)
            # img_name=f"{batch_idx}_iter_{iteration}_raw.png"
            # plt.save(img_data, img_folder+ img_name)
            
        elif batch_idx> max(img_idx_list):
            break
        else:
            pass
    return 


            
            
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
    parser.add_argument('--jet_color', type=str2bool, default=True)
    parser.add_argument('--use_pil_library', type=str2bool, default=True)


    parser.add_argument('--function', type=str, default='generate',choices=['generate','stitch'])
    
    

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device(args.cuda_dev if args.cuda else "cpu")
 
    
    
    
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
 
    # img_idx_list=[4,9,20,24,36,144,159,83,108,143,190,213]

    img_idx_list=[9, 143]


    
    
    testing_dataset = PASCAL_dataset(**prefixed_args,mode="test",full_supervison=True)
    test_loader = torch.utils.data.DataLoader(testing_dataset, batch_size=1)
    class_dict=testing_dataset.new_classes


    save_folder=f"../results/heatmaps/{lrp0_dict['name']}_{args.backbone}_lr{args.lr}_bs{args.batch_size}"

    ###############generate_figures###############
    if args.function== 'generate':
        model=intialize_model(args, lrp0_dict)
        #load the pretrain checkpoint
        checkpoint_name=f"../snapshot/"+ f"{args.backbone}_{args.pre_epochs}_pre_train_21.pth"
        checkpoint=torch.load(checkpoint_name)
        model.load_state_dict(checkpoint)
        model.to(device)
        model.eval()
        with torch.no_grad():
            # heatmap_plot(model, test_loader, lrp0_dict,device, iteration=156760, n_masks=0, save_folder=save_folder )
            heatmap_plot(model, test_loader, lrp0_dict,device, iteration=156760, n_masks=0, save_folder=save_folder )
            for  n_masks in lrp0_dict['list_n_masks']:
                for iteration in lrp0_dict['list_iter']:
                    model=load_checkpoint(args, model,lrp0_dict['name'], iteration, n_masks)
                    heatmap_plot(model, test_loader, lrp0_dict,device, iteration, n_masks,save_folder)
    else:
        ###############stitch the images together###############
        color_style="jet_color" if args.jet_color==True else "red_blue_color"

        
        ############ may need to cut some range


        for batch_idx, (data, class_labels, sem_gts, sem_gt_exists) in tqdm(enumerate(test_loader)):
            if batch_idx in img_idx_list:
                img_idx=batch_idx
                # FIX THE BATCHSIZE=1
                class_labels=class_labels[0] # C
                class_labels[0]=0 #shut down the background class
                exist_c_idx=np.nonzero(class_labels.numpy())[0] # picke the non zero element index along dim=0 #C 

                for k in exist_c_idx:
                   
                    for n_masks in lrp0_dict['list_n_masks']:

                        # draw one image
                        new_im = Image.new(mode="RGBA", size=(((1+len(lrp0_dict['list_iter']))*271 + 15) , 256+40),color="white")
                      
                        draw = ImageDraw.Draw(new_im)
            
                        ########################## first image should come from mask=0 case and iter 156760 :
                        iteration=156760
                        fig_path="../results/heatmaps/"+ f"{lrp0_dict['name']}_{args.backbone}_lr{args.lr}_bs{args.batch_size}/" + \
                                        f"{color_style}_lab0/"
                        path=fig_path+f"{img_idx}_iter_{iteration}_class_{class_dict[k]}.png"
                        im = Image.open(path)
                        new_im.paste(im,(15,10))
                        draw.text((135+ 12,256+12),f"{iteration}",fill=(0,0,0)) # black color

                        ###################### draw other images
                        for i ,iteration in enumerate(lrp0_dict['list_iter']):
                            
                            fig_path="../results/heatmaps/"+ f"{lrp0_dict['name']}_{args.backbone}_lr{args.lr}_bs{args.batch_size}/" + \
                                    f"{color_style}_lab{n_masks}/"
                            path=fig_path+f"{img_idx}_iter_{iteration}_class_{class_dict[k]}.png"

                            im = Image.open(path)
                            new_im.paste(im,((i+1)*271+ 15,10))
                            draw.text(((i+1)*271+135+ 12 ,256+12),f"{iteration}",fill=(0,0,0)) # black color


                     
                        os.makedirs(f"../results/heatmaps/lrp0_{args.backbone}_lr{args.lr}_bs{args.batch_size}/final/{color_style}/", exist_ok=True)
                        new_im.convert("RGB").save(f"../results/heatmaps/lrp0_{args.backbone}_lr{args.lr}_bs{args.batch_size}/final/{color_style}/lab{n_masks}_{img_idx}_iter_156760-{end}_class_{class_dict[k]}.png")
            else:
                pass
