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
    print(f"For model {name}, load the checkpoint{checkpoint_name} ")
    checkpoint=torch.load(checkpoint_name)
    model.load_state_dict(checkpoint)
    return model


def plot_fig_per_img(args,model,batch_idx, data,sem_gts,device, path_save, dict):
    num_cases=len(dict['list_n_masks'])
    cmap = color_map()
    pred_segs=[]
    for iteration, n_masks in zip(dict['list_iter'],dict['list_n_masks']):
        model=load_checkpoint(args, model, dict['name'],iteration, n_masks)
        # model.to(device)
        # model.eval()
        # predict heatmap
        if dict['name']=="std_unet":
            heatmaps = model(data)
        elif dict['name']=="mt_unet":
            heatmaps,_ = model(data)
        else:
            _ ,heatmaps= model(data)

        # infere classmaps via argmax
        class_maps = torch.argmax(heatmaps[0], dim=0) # H x W
        pred_seg = draw_colors(class_maps.cpu().numpy(),cmap)
        pred_segs.append(pred_seg)
    
        # draw image
        img_data = np.uint8(data[0].permute(1,2,0).cpu().numpy()*255)
        img_gt = draw_colors(sem_gts[0].cpu().numpy(),cmap)
        
    # make first combined plots for choosing good example
    fig, axs = plt.subplots(1, 2+num_cases, figsize=(14, 3), sharey=True)
    plt.subplots_adjust(wspace = 0.05)    
    axs[0].imshow(img_data)
    axs[0].axis('off')
    for i in range(num_cases):
        axs[1+i].imshow(pred_segs[i])
        axs[1+i].axis('off')
    axs[1+num_cases].imshow(img_gt)
    axs[1+num_cases].axis('off')


    # add y label
    axs[0].text(-0.1,0.5, dict['y_txt'], size=12, ha="center", 
                transform=axs[0].transAxes)


    if dict['add_xlabel']:
        axs[0].text(0.5,-0.1, 'Input', size=12, ha="center", 
                transform=axs[0].transAxes)
        for i , n_masks in enumerate(dict['list_n_masks']):
            if dict['list_n_masks'][i] is not None:
                xlabel_text=f'{n_masks} masks'
            else:
                xlabel_text=f'1464 masks'
            axs[i+1].text(0.5,-0.1, xlabel_text, size=12, ha="center", 
                transform=axs[i+1].transAxes)
        axs[1+num_cases].text(0.5,-0.1, 'GT', size=12, ha="center", 
                transform=axs[1+num_cases].transAxes)


    plt.savefig(path_save + '/' + 'overview_'+str(batch_idx)+'.png')

    # clear the pred_segs
    pred_segs.clear()

    # # save raw image of prediction with same arguments 
    # img = Image.fromarray(np.uint8(img*255))
    # img.save(path_save + '/' + 'pred_'+str(batch_idx)+'.png')

    # save gt and input
    # img_gt = Image.fromarray(np.uint8(img_gt*255))
    # img_data = Image.fromarray(img_data)
    # img_gt.save(path_save + '/' + 'gt_'+str(batch_idx)+'.png')
    # img_data.save(path_save + '/' + 'data_'+str(batch_idx)+'.png')
    plt.close(fig)
    

def plot_figs(args,testing_dataset,dict,img_idx_dict=None):
    # set the path_save
    if len(dict['list_n_masks'])>1:
        folder_name = f"{dict['name']}_{args.backbone}_lr{args.lr}_bs{args.batch_size}"
    else:
        folder_name = f"{dict['name']}_{args.backbone}_iter{dict['list_n_masks'][0]}_lab{dict['list_iter'][0]}_lr{args.lr}_bs{args.batch_size}"
    path_save = "../results/qualitative/" + folder_name  
    

    # initialize model according to dict['name']
    print(f"-----------initialize model {dict['name']}-----------------")
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
    model.eval()
    
    with torch.no_grad():
        test_loader = torch.utils.data.DataLoader(testing_dataset, batch_size=dict['test_batch_size'])
        for order_by,sub_img_dict in img_idx_dict.items():
            for key, img_idx_list in sub_img_dict.items():
                # generate the prediction 
                os.makedirs(path_save + f'/{order_by}_{key}', exist_ok=True)
                print(f"Created images will be save in the following path {path_save}/{order_by}_{key}")
                for batch_idx, (data, class_labels, sem_gts, sem_gt_exists) in tqdm(enumerate(test_loader)):
                    sem_gts = sem_gts.long().to(device)
                    data = data.to(device)
                    class_labels = class_labels.to(device)

                    if batch_idx is not None:
                        if batch_idx in img_idx_list:
                            plot_fig_per_img(args, model,batch_idx, data,sem_gts, device, path_save + f'/{order_by}_{key}', dict)
                        elif batch_idx> max(img_idx_list):
                            break
                        else:
                            pass

                    else: # otherwise go through all test images 1449
                        plot_fig_per_img(args, model,batch_idx, data,sem_gts,device, path_save + f'/{order_by}_{key}', dict)
            
            
            
    # set colorscheme for 20 classes TODO make disjoint appearing colors for co-present classes
    

    
    # colors = ["#ffffff",
    #         "#d1d1d1", 
    #         "#818181",
    #         "#ff2400",
    #         "#f72685", 
    #         "#fe74fe",
    #         "#7206b6", 
    #         "#380da4", 
    #         "#4461ee", 
    #         "#50c9f0",  
    #         "#fd8f2f", 
    #         "#f4d403", 
    #         "#50e316", 
    #         "#1d6d1f", 
    #         "#e89ff0",  
    #         "#bcf1a8", 
    #         "#87a9fd", 
    #         "#c4f418",  
    #         "#5faf66", 
    #         "#f68d92", 
    #         "#63fea6"]

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
    # parser.add_argument('--name', type=str, default="std_unet",choices=['std_unet','mt_unet','lrp0'])
    parser.add_argument('--backbone', type=str, default='resnet50',choices=['vgg16','vgg16_bn','resnet18','resnet34','resnet50','resnet101'], help='the backbone for the encoder')
    parser.add_argument('--iteration', type=int, default=0,help="related to loaded checkpoint name")
    parser.add_argument('--pre_epochs', type=int, default=10,help="related to loaded checkpoint name")
    parser.add_argument('--no-cuda', action='store_true', default=False, help='enables CUDA training')

    # train
    parser.add_argument('--batch-size', type=int, default=10, help="related to loaded checkpoint name")
    parser.add_argument('--checkpoint_folder', type=str, default='../snapshot/', help='examples:"snapshot/", "/fast/AG_Kainmueller/xyu/"') #checkpoint_folder should be "snapshot/" or "/fast/AG_Kainmueller/xyu/"

    # loss desgin
    parser.add_argument('--lr', type=float, default=1e-5, metavar='N', help='learning rate')
    parser.add_argument('--results_path', type=str, default='results/', metavar='N', help='Where to store images')
    parser.add_argument('--cuda-dev', type=str, default='cuda', metavar='N')
    
    
    

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device(args.cuda_dev if args.cuda else "cpu")

    # ## import model
    # print("-----------import model-----------------")
    # if args.name == 'std_unet':
    #     args.add_classification=False
    #     model = MTUNet(3, args.backbone,  args.num_classes, args.add_classification)
    # elif args.name == 'mt_unet':
    #     args.add_classification=True
    #     model = MTUNet(3, args.backbone,  args.num_classes, args.add_classification)
    # elif args.name == 'lrp0':
    #     args.add_classification = True
    #     args.use_old_xai = False
    #     args.xai = 'LRP_epsilon'
    #     args.epsilon = 1e-8
    #     args.alpha = 1
    #     args.memory_efficient = False
    #     args.detach_bias = True
    #     model = Network(args)
    # else:
    #     raise NotImplementedError

    # args.cuda = not args.no_cuda and torch.cuda.is_available()
    # device = torch.device(args.cuda_dev if args.cuda else "cpu")
    # model.to(device)

    #     print(model)
    
    
    
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
    std_unet_dict={
        'name':'std_unet',
        'list_iter':[202500,252000,259500,304500] , #change TODO
        'list_n_masks':[20,100,500,None],
        'test_batch_size':1,
        'add_xlabel':False,
        'y_txt': '(a)'
    }
    mt_unet_dict={
        'name':'mt_unet',
        'list_iter':[207000,285000,432000,483000], #change TODO
        'list_n_masks':[20,100,500,None],
        'test_batch_size':1,
        'add_xlabel':False,
        'y_txt': '(b)'
    }
    lrp0_dict={
        'name':'lrp0',
        'list_iter':[198000,249000,246000,285000], #change TODO
        'list_n_masks':[20,100,500,None],
        'test_batch_size':1,
        'add_xlabel':True,
        'y_txt': '(c)'
    }
    # img_idx_list=[4,9,20,24,36,144,159,83,108,143,190,213]

    img_idx_dict={
        'head':{
         
            'test2':[9,20,144],
        #    'lrp0_20-std_unet_20': [1348, 1295, 527, 249, 1400, 615, 241, 1327, 1365, 1028, 1246, 1110, 7, 362, 1389, 298, 767, 1014, 341, 1304],
        #     'lrp0_20-mt_unet_20':[1348, 1327, 1365, 527, 249, 1028, 125, 615, 241, 1295, 263, 362, 350, 1014, 1389, 1400, 298, 624, 1339, 1246],
        #     'lrp0_100-std_unet_100': [617, 463, 1337, 1340, 1066, 1303, 1145, 39, 799, 688, 1092, 495, 1035, 1009, 1118, 1293, 229, 1313, 1081, 1414],
        #     'lrp0_100-mt_unet_100': [795, 1340, 116, 779, 933, 274, 1257, 362, 723, 1153, 8, 87, 1404, 776, 1060, 1145, 709, 1121, 1098, 1136]
        },
        # 'tail':{
        #     'lrp0_20-std_unet_20': [824, 324, 970, 274, 153, 911, 126, 813, 927, 610, 607, 535, 974, 579, 679, 792, 706, 1193, 237, 961],
        #     'lrp0_20-mt_unet_20': [556, 485, 5, 500, 324, 607, 354, 328, 1346, 688, 1441, 1237, 813, 1284, 579, 494, 761, 1193, 706, 237],
        #     'lrp0_100-std_unet_100': [441, 785, 745, 589, 674, 1436, 168, 533, 160, 1312, 1226, 84, 778, 796, 858, 786, 980, 74, 298, 1394],
        #     'lrp0_100-mt_unet_100':[395, 682, 103, 1352, 236, 356, 200, 400, 420, 357, 980, 383, 373, 346, 1432, 408, 1293, 744, 407, 430]
        # }

    }
    # start to generate figs
    testing_dataset = PASCAL_dataset(**prefixed_args,mode="test",full_supervison=True)
    plot_figs(args,testing_dataset,std_unet_dict,img_idx_dict)
    plot_figs(args,testing_dataset,mt_unet_dict,img_idx_dict)
    plot_figs(args,testing_dataset,lrp0_dict,img_idx_dict)



    # stitch the subfigs for three models
    def resize_im(path, left,right, top, bottom):
        im = np.array(Image.open(path))[top:-bottom,left:-right,:]
        im = Image.fromarray(im)
        return im
    
   

    for order_by,sub_img_dict in img_idx_dict.items():
        for key, img_idx_list in sub_img_dict.items():
            for img_idx in img_idx_list:
                new_im = Image.new(mode="RGBA", size=(1130,595))
                # new_im = Image.new(mode="RGBA", size=(1000,1130))
                for i, name in enumerate(['std_unet','mt_unet']):
                    fig_path="../results/qualitative/"+ f"{name}_{args.backbone}_lr{args.lr}_bs{args.batch_size}/"+ f'{order_by}_{key}/'
                    path = fig_path + f'overview_{img_idx}.png'
                    im=resize_im(path, 140,130,55,55)
                    new_im.paste(im,(0,190*i))
                    
                # bottom img
                fig_path="../results/qualitative/"+f"lrp0_{args.backbone}_lr{args.lr}_bs{args.batch_size}/"+ f'{order_by}_{key}/'
                path = fig_path + f'overview_{img_idx}.png'
                im=resize_im(path, 140,130,50,35)
                new_im.paste(im,(0,190*2))
                # new_im.show()
                
                os.makedirs(f"../results/qualitative/final/{order_by}_{key}", exist_ok=True)
                new_im.save(f"../results/qualitative/final/{order_by}_{key}/{img_idx}.png")

   


   

    


   
    



    

  
    

   