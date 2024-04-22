import torch
from bs4 import BeautifulSoup 
import numpy as np
import os
import sys
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from utils.sampling_utils import *
import random



class PASCAL(object):
    def __init__(self,args):

        random.seed(100)
        self.prefixed_args={"path": args.data_path, 
                            "seed": args.seed,
                            "num_labels": args.num_labels,
                            "uniform_masks": args.uniform_masks,
                            "semisup_dataset": args.semisup_dataset,
                            "crop_size": args.crop_size
                            }

        # if use unet model, train the model in full supervison way
        if args.semisup_dataset:
            self.training_dataset = PASCAL_dataset(**self.prefixed_args,mode="train",full_supervison=False)
        else:
            self.training_dataset = PASCAL_dataset(**self.prefixed_args,mode="train",full_supervison=True)
        self.testing_dataset = PASCAL_dataset(**self.prefixed_args,mode="test",full_supervison=True)

        assert args.num_classes== self.training_dataset.num_classes,"number in classes in dataset.py file and the num_classes parameter you set"
        
        if args.pre_batch_size is not None and args.pre_epochs>0:
            # self.prefixed_args["weighted_sampling"] = False
            self.pre_training_dataset = PASCAL_dataset(**self.prefixed_args,mode="train",full_supervison=False) #TODO check
            self.pre_testing_dataset = PASCAL_dataset(**self.prefixed_args,mode="test",full_supervison=True)

        


class PASCAL_dataset(torch.utils.data.Dataset):
    def __init__(self, **kwargs):
        
        self.path= kwargs['path'] + 'VOC2012/' 
        self.folder_images = 'JPEGImages/'
        self.folder_labels = 'Annotations/'
        self.folder_masks = 'SegmentationClass/'
        
        self.dataset = 'full'
        self.mode=kwargs['mode']
        self.num_labels = kwargs['num_labels']
        self.uniform_masks = kwargs['uniform_masks']
        self.crop_size = kwargs['crop_size']
        
        self.set_classes()
        self.img_names,self.lab_names,self.mask_names,self.idx_mask,self.idx_no_mask = self.load_names()
        self.num_classes = len(self.classes)

        self.full_supervison=kwargs['full_supervison']

    
    def set_classes(self):
        # The segmentation part of the PASCAL VOC 2012 dataset contains 22 classes
        self.classes = {0:'bg',1: 'aeroplane',2:'bicycle',3:'bird',4:'boat',5:'bottle',6:'bus',
            7:'car',8:'cat',9:'chair',10:'cow',11:'diningtable',12:'dog',
            13:'horse',14:'motorbike',15:'person',16:'pottedplant',17:'sheep',18:'sofa',19:'train',
            20:'tvmonitor'}
        self.special_class = {255:'boundary'}
        self.class_encoding = np.fromiter(self.classes.keys(),dtype=np.uint8)


    def load_names(self):
        # get the image and label names of the classification dataset
        img_names = sorted(os.listdir(self.path+self.folder_images))
        lab_names = sorted(os.listdir(self.path+self.folder_labels))

        # get the subset of segmentation images (without extension)
        mask_names = sorted(os.listdir(self.path+self.folder_masks))
        mask_names_val = self.read_list(self.path+'ImageSets/Segmentation/val.txt')

        # split the dataset 
        split_img_names = self.get_subset(img_names,mask_names_val)
        split_lab_names = self.get_subset(lab_names,mask_names_val)
        split_mask_names = self.get_subset(mask_names,mask_names_val)

        if self.mode=="train":
            sub_img_names = split_img_names[0]
            sub_lab_names = split_lab_names[0]
            
            # reduce number of training masks
            # old version: reduced_mask_names = random_subset(split_mask_names[0],self.fraction_masks) 
            if self.num_labels != None:
                reduced_mask_names = self.controlled_random_mask_sampling(mask_names=split_mask_names[0],all_lab_names=sub_lab_names,
                                                                          num_labels=self.num_labels,alpha_uniform=self.uniform_masks)
            else:
                reduced_mask_names = split_mask_names[0]
            
            sub_mask_names,idx_mask,idx_no_mask = self.check_mask_exist(sub_img_names,reduced_mask_names)

            print(f'number of train images / number of images: {len(sub_mask_names)} / {len(img_names)}')
            print(f'number of train images with masks / number of train images: {len(idx_mask)} / {len(sub_mask_names)}')
            print(f'number of images without masks / number of train images: {len(idx_no_mask)} / {len(sub_mask_names)}')
            print('\n')
  
        elif self.mode=="val" or self.mode=="test":
            sub_img_names = split_img_names[1]
            sub_lab_names = split_lab_names[1]
            sub_mask_names,idx_mask,idx_no_mask = self.check_mask_exist(sub_img_names,mask_names)
        else:
            raise NotImplementedError("The mode parameter can only be 'train', 'val' and 'test'.")

        return sub_img_names,sub_lab_names,sub_mask_names,idx_mask,idx_no_mask




    def controlled_random_mask_sampling(self,mask_names,all_lab_names,num_labels,alpha_uniform,exclude_bg=True,seed = 0):
        
        lab_names = self.get_subset(all_lab_names,self.remove_extension(mask_names))[1]
        labs = self.load_class_labels(lab_names)
        if exclude_bg:
            labs = labs[:,1:]
        
        print(len(mask_names))
        print(mask_names[0])
        print(len(lab_names))
        print(lab_names[0])
        # print(labs)
        
        dist = compute_distribution(labs_one_hot=labs)
        dist = make_uniform(dist,alpha_uniform)
        sample_ids = random_sampling_with_distribution(labs,num_labels,dist,seed=seed)
        reduced_mask_names = list(np.array(mask_names)[sample_ids])

        return reduced_mask_names


   

    def read_list(self,path):
        with open(path, "r") as f:
            file_list = f.read().split('\n')
        f.close()
        return file_list[:-1]


    def remove_extension(self,list):
        new_list = []
        for item in list:
            new_list.append(item.split(".")[0])

        return new_list


    def get_key(self,values,my_dict):
        keys = []
        for key, value in my_dict.items():
            for val in values:
                if val == value:
                    keys.append(key)
        return keys


    def get_subset(self,list,subset):
        '''
        list is a list with file names with extensions 
        subset is a list of file names without extensions
        returns the subtraction of subset from list and the intersection
        '''
        inter_list = []
        remain_list = []
        for item in list:
            if item.split(".")[0] in subset:
                inter_list.append(item)
            else:
                remain_list.append(item)
        return remain_list, inter_list


    def check_mask_exist(self,img_names,mask_names):
        '''
        checks if a segmetnation masks exists
        '''
        mask_list = []
        idx_mask = []
        idx_no_mask = []
        for i,item in enumerate(img_names):
            # remove img extension and add mask extension
            mask_name = item.split(".")[0] + '.' +mask_names[0].split(".")[1]

            if mask_name in mask_names:
                mask_list.append(mask_name)
                idx_mask.append(i)
            else:
                idx_no_mask.append(i)
                mask_list.append('None')

        return mask_list,idx_mask,idx_no_mask


    def get_statistics(self):

        counts = np.zeros(len(self.class_encoding))
        n_samples = len(self.lab_names)
        for i in range(n_samples):
            lab = self.get_class_label(self.path+self.folder_labels+self.lab_names[i],self.classes)
            lab_one_hot = self.one_hot(lab,self.class_encoding)
            counts += lab_one_hot
        counts[0] = n_samples # bg always present

        return counts,n_samples


    def load_one_hot_class_label(self,lab_name):
        
        lab = self.get_class_label(self.path+self.folder_labels+lab_name,self.classes) # TODO something up in here
        lab = np.unique(lab) # ensure that there are no doubles in label
        lab = self.one_hot(lab,self.class_encoding)
        lab[0] = 1

        return lab

    
    def load_class_labels(self,list_labels):
        
        labs = np.array([[]])

        for i,lab_name in enumerate(list_labels):
            lab = self.load_one_hot_class_label(lab_name)
            labs = np.append(labs,[lab],axis=int(i==0))

        return labs


    def __len__(self):
        if self.full_supervison:
            return len(self.idx_mask)
        else:
            return len(self.img_names)


    def __getitem__(self,idx):
              
        # load image and lab
        if self.full_supervison:
            idx=self.idx_mask[idx]
            img = np.array(Image.open(self.path+self.folder_images+self.img_names[idx]))
            lab = self.load_one_hot_class_label(self.lab_names[idx])
        else:
            img = np.array(Image.open(self.path+self.folder_images+self.img_names[idx]))  
            lab = self.load_one_hot_class_label(self.lab_names[idx])

        # check if img with idx has a mask 
        if self.mask_names[idx] != 'None':
            mask = np.array(Image.open(self.path+self.folder_masks+self.mask_names[idx]))
            mask[mask == 255] = 0 # delete boundary class
            mask_exist = True
        else:
            mask = -1+0*np.copy(img[:,:,0])
            mask_exist = False

        # augmentation
        small_side = min(img.shape[:-1])
        if self.mode=="train":
            transform = A.Compose([
                A.RandomCrop(height=small_side,width=small_side,always_apply=True),
                A.Resize(height=self.crop_size,width=self.crop_size),
                A.Defocus(radius=(1,5)),
                A.RandomBrightnessContrast(p=2),
                A.ColorJitter(),
                A.HorizontalFlip(p=0.5),
                A.GaussNoise(),
                ToTensorV2()
            ])
          
        else:
            transform = A.Compose([
                A.CenterCrop(height=small_side,width=small_side,always_apply=True),
                A.Resize(height=self.crop_size,width=self.crop_size),
                ToTensorV2()])
        
        transformed = transform(image=img,mask=mask) 
        mask = transformed["mask"] # HxW
        mask = mask.type(torch.float32)
        img = transformed["image"] # HxWxC -> CxHxW
        img = ((img.type(torch.float32)+1)/256.0).type(torch.float32) 
    

        # get label from transformed mask -> only if mask exists
        if mask_exist:
            lab = self.get_label_from_mask(mask) # replace the one_hot function
            lab[0] = 1 # not important just for consistency

        return img, lab, mask, mask_exist
       
   

    def get_class_label(self,path_xml,classes):
        
        with open(path_xml, 'r') as f:
            data = f.read() 

        # pass to bs parser and find all object entries
        bs_data = BeautifulSoup(data, 'xml') 
        b_names = bs_data.find_all('name')

        # extract all the seen classes
        img_label = []
        for j in b_names:
            img_label = img_label+self.get_key([j.string],classes)

        return np.unique(img_label)


    def one_hot(self,lab,class_encoding):
        '''
            converts a class label [4,5,10] or [4,5,5,5,10]
            into an one-hot encoding [0,0,1,0,0,..] depending on the class encoding [0,1,2,..]
        '''
        lab_one_hot = sum([class_encoding == i for i in lab])
        lab_one_hot[lab_one_hot > 1] = 1
        
        return lab_one_hot

    
    def get_label_from_mask(self,mask):
        '''
        mask shape is [H,W] H=W=256 after cropping
        '''
        label=np.array( [(mask==i).sum() for i in range(self.num_classes)])
        label[label>0]=1
        return label.astype(np.uint8)





