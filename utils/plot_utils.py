import numpy as np
import torch
from PIL import Image, ImageDraw
from skimage import color

import sys
# --------------------------------------
# Plot functions
# --------------------------------------
# colorsList = [(0,0,0),(0,255,0),(255,255,255),(0,255,255),(0,0,255),(255,0,0),(125,125,0)] #0-black,1-lime,2- white,3-cyan,4-blue,5-red,6-olive 
colorsList= [(0,0,0),(0,255,0),(255,255,255),(0,255,255),(0,0,255),(255,0,0),(125,125,0),(1,40,20),(1,80,20),(20,40,0),(50,200,0),(0,200,90),(30,60,100),(100,20,60),
(100,200,0),(200,20,60),(50,20,100),(22,34,180),(220,22,0),(40,200,30),(140,26,70),(70,26,26)]
def color_sem_gt(X):
    image = Image.fromarray(X, mode='L')
    sh = X.shape
    
    image2 = Image.new(mode='RGB', size=(sh[1],sh[0])) # PIL(W,H) and numpy (H,W) print the shape in different way
    draw = image.load()
    X = X.astype(np.int32)
    for w in range(sh[1]):
        for h in range(sh[0]):
            if X[w,h]==255:
                image2.putpixel((h, w), tuple((122,35,45)))
            else:
                image2.putpixel((h, w), tuple(colorsList[X[w,h]]))
    return image2


def show_raw_sem_gt(raws, sem_gts, labels):
    '''
    raw,sem_gt, on cpu torch.tensor
    '''
    # create a image
    new_image=Image.new('L',(56*8,28*16))
    for i in range(8):
        for j in range(16):
            idx=i*8+j
            raw=raws[idx,0,:,:]
            sem_gt=sem_gts[idx,0,:,:]

            new_image.paste(Image.fromarray(raw.numpy()*255),(56*i,28*j))
            new_image.paste(Image.fromarray(sem_gt.numpy()*255),(28+56*i,28*j))
    new_image.show()


def normalize_image_percentile(in_img):
    img_max=np.percentile(in_img,99.98)
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
    sat = normalize_image_percentile(np.abs(img))
    # sat = np.abs(img)

    brightness = np.ones_like(img) # add saturation, otherwise very dark pixels do not get colored

    blended_hsv = np.stack([hue,sat, brightness], axis=2)
    blended_hsv_rgb = np.round(color.hsv2rgb(blended_hsv)*255).astype(np.uint8)/255
    return blended_hsv_rgb 
    
    return

def save_ori_raw_raw_sem_gt(ori_raws,raws,sem_gts, title , save_folder, color=False):
    '''
    draw raw--predicted heatmap--sem_gt

    raw: [N,C,H,W]
    heatmaps: [N,num_classes, H, W]
    sem_gts: [N, H, W]
    '''

    new_image=Image.new('RGB',(768*2,256*5))
    draw = ImageDraw.Draw(new_image)
    for i in range(2):
        for j in range(5):
            idx=i*5+j
            if idx >= raws.shape[0]:
                pass
            else:
                
                ori_raw=ori_raws[idx,:,:,:].detach().cpu().numpy()
                raw=np.transpose(raws[idx,:,:,:].detach().cpu().numpy(),(1,2,0))
               
               
                sem_gt=sem_gts[idx,:,:].detach().cpu().numpy()

                new_image.paste(Image.fromarray((ori_raw*255).astype(np.uint8)),(768*i,256*j))
                new_image.paste(Image.fromarray((raw*255).astype(np.uint8)),(256+768*i,256*j))             
                new_image.paste(color_sem_gt(sem_gt),(512+768*i,256*j))

    new_image.save(save_folder+title)

std=np.array([0.229, 0.224, 0.225])
mean=np.array([0.485, 0.456, 0.406])

    
def save_raw_heatmap_sem_gt(rows, columns,raws,heatmaps,sem_gts, title , save_folder, color=False, normalize=False):
    '''
    draw raw--predicted heatmap--sem_gt

    raw: [N,C,H,W]
    heatmaps: [N,num_classes, H, W]
    sem_gts: [N, H, W]
    '''
    
    h,w=raws.shape[-2],raws.shape[-1]
    new_image=Image.new('RGB',(w*3*columns,h*rows))
    draw = ImageDraw.Draw(new_image)
    for i in range(columns):
        for j in range(rows):
            idx=i*rows+j
            if idx >= raws.shape[0]:
                pass
            else:
                raw=np.transpose(raws[idx,:,:,:].detach().cpu().numpy(),(1,2,0))
                pred_heatmap= np.argmax(heatmaps[idx,:,:,:].detach().cpu().numpy(),axis=0)
             
                sem_gt=sem_gts[idx,:,:].detach().cpu().numpy()

                # if normalize before, now you need to do inverse operation
                if normalize:
                    raw= std[np.newaxis,np.newaxis,:]*raw+mean[np.newaxis,np.newaxis,:]
                new_image.paste(Image.fromarray((raw*255).astype(np.uint8)),(768*i,256*j))
                new_image.paste(color_sem_gt(pred_heatmap),(256+768*i,256*j))
                new_image.paste(color_sem_gt(sem_gt),(512+768*i,256*j))

    new_image.save(save_folder+title)

def save_raw_confidencemap_sem_gt(rows, columns,raws,heatmaps,sem_gts, title , save_folder, color=False, normalize=False):
    '''
    draw raw--predicted heatmap--sem_gt

    raw: [N,C,H,W]
    confidence_maps: [N, H, W]
    sem_gts: [N, H, W]
    '''
    pseudo_logits, pseudo_labels = torch.max(torch.softmax(heatmaps.detach().cpu(), dim=1), dim=1)
    h,w=raws.shape[-2],raws.shape[-1]
    new_image=Image.new('RGB',(w*4*columns,h*rows))
    draw = ImageDraw.Draw(new_image)
    for i in range(columns):
        for j in range(rows):
            idx=i*rows+j
            if idx >= raws.shape[0]:
                pass
            else:
                raw=np.transpose(raws[idx,:,:,:].detach().cpu().numpy(),(1,2,0))
                confidence_map=pseudo_logits[idx,:,:].numpy()-0.5
                confident_region_map=((pseudo_logits[idx,:,:].ge(0.9))*pseudo_labels[idx,:,:]).numpy()
                sem_gt=sem_gts[idx,:,:].detach().cpu().numpy()

                # if normalize before, now you need to do inverse operation
                if normalize:
                    raw= std[np.newaxis,np.newaxis,:]*raw+mean[np.newaxis,np.newaxis,:]
                new_image.paste(Image.fromarray((raw*255).astype(np.uint8)),(256*4*i,256*j))
                new_image.paste(Image.fromarray((color_heatmap(confidence_map)*255).astype(np.uint8)),(256+256*4*i,256*j))
                new_image.paste(color_sem_gt(confident_region_map),(512+256*4*i,256*j))
                new_image.paste(color_sem_gt(sem_gt),(768+256*4*i,256*j))

    new_image.save(save_folder+title)

def save_heatmap_sem_gt(rows, columns, gtl_heatmaps,pl_heatmaps,sem_gts, corrects, title , save_folder, color=False):
    '''
    draw predicted label heatmap--GT label heatmap--sem_gt
    '''
    h,w=sem_gts.shape[-2],sem_gts.shape[-1]
    new_image=Image.new('RGB',(w*3*columns,h*rows))
    draw = ImageDraw.Draw(new_image)
    for i in range(columns):
        for j in range(rows):
            idx=i*rows+j
            if idx>=sem_gts.shape[0]:
                pass
            else:
            
                # if color the heatmaps
                if color:
                    gtl_heatmap=gtl_heatmaps[idx,0,:,:].detach().cpu().numpy()
                    gtl_heatmap=color_heatmap(gtl_heatmap)

                    pl_heatmap=pl_heatmaps[idx,0,:,:].detach().cpu().numpy()
                    pl_heatmap=color_heatmap(pl_heatmap)
                else:
                    gtl_heatmap=gtl_heatmaps[idx,0,:,:].detach().cpu().numpy()
                    gtl_heatmap[gtl_heatmap<0]=0
                    gtl_heatmap=gtl_heatmap/gtl_heatmap.max()

                    pl_heatmap=pl_heatmaps[idx,0,:,:].detach().cpu().numpy()
                    pl_heatmap[pl_heatmap<0]=0
                    pl_heatmap=pl_heatmap/pl_heatmap.max()


                sem_gt=sem_gts[idx,0,:,:].detach().cpu().numpy()

                new_image.paste(Image.fromarray((pl_heatmap*255).astype(np.uint8)),(84*i,28*j))
                new_image.paste(Image.fromarray((gtl_heatmap*255).astype(np.uint8)),(28+84*i,28*j))
                new_image.paste(Image.fromarray((sem_gt*255).astype(np.uint8)),(56+84*i,28*j))

                if corrects[idx]==0:
                    draw.line([(84*i,28*j+5),(84*i+28,28*j+5)], fill=128,width=3)

    new_image.save(save_folder+title)

