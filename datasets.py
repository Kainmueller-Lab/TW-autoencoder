import torch
from torchvision import datasets, transforms

from torchvision.transforms.transforms import RandomApply, GaussianBlur, ColorJitter
from utils.color_conversion import Rgb2Hed, Hed2Rgb, LinearContrast
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
from bs4 import BeautifulSoup 
from imgaug import augmenters as iaa
from collections import defaultdict
import copy
import numpy as np
import os
import csv
import sys
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from utils.sampling_utils import *
import random
import matplotlib.pyplot as plt



### specific Dataloader




### Datasets
class MNIST(object):
    def __init__(self, args):
        kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
        self.train_loader = torch.utils.data.DataLoader(
            datasets.MNIST('data/mnist', train=True, download=True,
                           transform=transforms.ToTensor()),
            batch_size=args.batch_size, shuffle=True, **kwargs)
        self.test_loader = torch.utils.data.DataLoader(
            datasets.MNIST('data/mnist', train=False, transform=transforms.ToTensor()),
            batch_size=args.batch_size, shuffle=True, **kwargs)





class EMNIST(object):
    def __init__(self, args):
        kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
        self.train_loader = torch.utils.data.DataLoader(
            datasets.EMNIST('data/emnist', train=True, download=True, split='byclass',
                           transform=transforms.ToTensor()),
            batch_size=args.batch_size, shuffle=True, **kwargs)
        self.test_loader = torch.utils.data.DataLoader(
            datasets.EMNIST('data/emnist', train=False, split='byclass',
            transform=transforms.ToTensor()),
            batch_size=args.batch_size, shuffle=True, **kwargs)





class FashionMNIST(object):
    def __init__(self, args):
        kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
        self.train_loader = torch.utils.data.DataLoader(
            datasets.FashionMNIST('data/fmnist', train=True, download=True,
                           transform=transforms.ToTensor()),
            batch_size=args.batch_size, shuffle=True, **kwargs)
        self.test_loader = torch.utils.data.DataLoader(
            datasets.FashionMNIST('data/fmnist', train=False, transform=transforms.ToTensor()),
            batch_size=args.batch_size, shuffle=True, **kwargs)





class Lizard(object):
    def __init__(self,args):
        self.prefixed_args={"path": args.data_path, 
                            "seed": args.seed,
                            "classify_class_label_list":[1,2,3,4,5,6],
                            "basic_aug_params":{
                                                'fliplr': {'prob': 0.5},
                                                'flipud': {'prob': 0.5},
                                                'rotate90': {'start_factor': 1, 'end_factor': 4},
                                                'gammacontrast': {'gamma':(0.5,2.0)},
                                                'elastic': {'alpha': (0,70.0),'sigma':10.0 }
                                                },
	                        "color_aug_params":0.5
                            }

        assert args.num_classes== len(self.prefixed_args["classify_class_label_list"]),"please check the self.prefixed_args in dataset.py file and the num_classes parameter you set"
        
        if args.num_classes==7:
            self.prefixed_args["classify_class_label_list"]=[0,1,2,3,4,5,6]

        if args.uniform_class_sampling:
            print("use uniform class sampling")
            lizaddataset=Lizard_dataset(**self.prefixed_args,mode="train")
            sampling_weights=lizaddataset.get_class_weights_all()
            sampler = torch.utils.data.WeightedRandomSampler(sampling_weights, num_samples=len(sampling_weights), replacement=True,generator=None)
            
            self.train_loader=torch.utils.data.DataLoader(lizaddataset,
                                                batch_size=args.batch_size,shuffle=False, sampler=sampler)
        else:
            self.train_loader=torch.utils.data.DataLoader(Lizard_dataset(**self.prefixed_args,mode="train"),
                                                batch_size=args.batch_size,shuffle=True)

        # test dataset
        self.test_loader=torch.utils.data.DataLoader(Lizard_dataset(**self.prefixed_args,mode="test"),
                                                batch_size=args.batch_size, shuffle=False)

        if args.pre_batch_size is not None and args.pre_epochs>0:
            self.pre_train_loader=torch.utils.data.DataLoader(Lizard_dataset(**self.prefixed_args,mode="train"),
                                            batch_size=args.pre_batch_size,shuffle=True)
            self.pre_test_loader=torch.utils.data.DataLoader(Lizard_dataset(**self.prefixed_args,mode="test"),
                                                batch_size=args.pre_batch_size, shuffle=False)




class PASCAL(object):
    def __init__(self,args):

        random.seed(100)
        self.prefixed_args={"path": args.data_path, 
                            "seed": args.seed,
                            "n_masks": args.n_masks,
                            "uniform_masks": args.uniform_masks,
                            "reduce_classes":args.reduce_classes,
                            "weighted_sampling": args.weighted_mask_sampling,
                            "normalize": args.normalize,
                            "trafo_mode": args.trafo_mode
                            }

        if args.uniform_class_sampling:
            print("uniform sampling not supported")
        
        if args.normalize:
            print("input image will be normalized")  

        # if use unet model, train the model in full supervison way
        if args.model=="AE":
            args.add_classification=True
        train_full_supervison=True if args.add_classification==False else False
   

        self.training_dataset = PASCAL_dataset(**self.prefixed_args,mode="train",full_supervison=train_full_supervison)
        self.testing_dataset = PASCAL_dataset(**self.prefixed_args,mode="test",full_supervison=True)

        assert args.num_classes== self.training_dataset.num_classes,"number in classes in dataset.py file and the num_classes parameter you set"
        
        if args.pre_batch_size is not None and args.pre_epochs>0:
            self.prefixed_args["weighted_sampling"] = False
            self.pre_training_dataset = PASCAL_dataset(**self.prefixed_args,mode="train",full_supervison=train_full_supervison)
            self.pre_testing_dataset = PASCAL_dataset(**self.prefixed_args,mode="test",full_supervison=True)

        







class PASCAL_dataset(torch.utils.data.Dataset):
    def __init__(self, **kwargs):
        
        self.path= kwargs['path'] + 'VOC2012/' 
        self.folder_images = 'JPEGImages/'
        self.folder_labels = 'Annotations/'
        self.folder_masks = 'SegmentationClass/'
        
        self.dataset = 'full'
        self.mode=kwargs['mode']
        self.n_masks = kwargs['n_masks']
        self.uniform_masks = kwargs['uniform_masks']
        self.return_img_idx=False
        self.trafo_mode = kwargs['trafo_mode']
        
        self.set_classes()
        self.define_new_classes(switched_on=kwargs["reduce_classes"])
        self.img_names,self.lab_names,self.mask_names,self.idx_mask,self.idx_no_mask = self.load_names()
        self.num_classes = len(self.new_classes)

        self.full_supervison=kwargs['full_supervison']
        self.normalization=kwargs['normalize']

    
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
            if self.n_masks != None:
                reduced_mask_names = self.controlled_random_mask_sampling(mask_names=split_mask_names[0],all_lab_names=sub_lab_names,
                                                                          n_masks=self.n_masks,alpha_uniform=self.uniform_masks)
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

    def weight_from_seg_gt(self):
        weight=np.zeros(self.num_classes)
        for i in range(len(self.idx_mask)):
            idx=self.idx_mask[i]
            if self.mask_names[idx] != 'None':
                mask = np.array(Image.open(self.path+self.folder_masks+self.mask_names[idx]))
                img = np.array(Image.open(self.path+self.folder_images+self.img_names[idx]))
                mask[mask == 255] = 0 # delete boundary class
                small_side = min(img.shape[:-1])
                if self.trafo_mode == 0:
                    transform = A.Compose([
                        A.RandomCrop(height=small_side,width=small_side,always_apply=True),
                        A.Resize(height=256,width=256),
                        A.Defocus(radius=(1,5)),
                        A.RandomBrightnessContrast(p=2),
                        A.ColorJitter(),
                        A.HorizontalFlip(p=0.5),
                        A.GaussNoise(),
                        ToTensorV2()
                    ])
                else:
                        transform = A.Compose([
                            A.RandomResizedCrop(height=256,width=256,always_apply=True,scale=(0.01, 1.0),ratio=(1,1)),
                            A.Defocus(radius=(1,4)),
                            A.RandomBrightnessContrast(p=2),
                            A.HorizontalFlip(p=0.5),
                            A.GaussNoise(),
                            ToTensorV2()
                        ])
                transformed = transform(image=img,mask=mask) 
                mask = transformed["mask"] # HxW
                # map to new rough or subset classes (identety map when turned of)
                mask = self.map_classes(mask,self.map)
                tmp_weight=np.bincount(mask.flatten(),minlength=self.num_classes)
                weight+=tmp_weight
        # print(f"There are {len(self.idx_mask)} masks with num_pixels {weight}")
        return torch.tensor(weight)



    def define_new_classes(self,switched_on = False):
        
        self.map = {}
        self.new_classes = self.classes
        self.new_class_encoding = self.class_encoding
        
        if switched_on == True:
            self.new_classes = {}
            new_classes_map = {'bg':['bg'],
                                'aeroplane':['aeroplane'],
                                'animal':['bird','cat','cow','dog','horse','sheep'],
                                'ground_vehicle':['bicycle','boat','bus','car','motorbike','train'],
                                'tvmonitor':['tvmonitor'],
                                'sofa_chair_table':['sofa','chair','diningtable'],
                                'person':['person']}

            for idx_new,new_class in enumerate(new_classes_map):
                for c in new_classes_map[new_class]:
                    self.map[self.get_key([c],self.classes)[0]] = idx_new
                self.new_classes[idx_new] = new_class

            for idx_old in self.classes.keys():
                if idx_old not in self.map.keys():
                    self.map[idx_old] = 0

            self.new_class_encoding = np.fromiter(self.new_classes.keys(),dtype=np.uint8)


    def controlled_random_mask_sampling(self,mask_names,all_lab_names,n_masks,alpha_uniform,exclude_bg=True,seed = 0):
        
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
        sample_ids = random_sampling_with_distribution(labs,n_masks,dist,seed=seed)
        reduced_mask_names = list(np.array(mask_names)[sample_ids])

        return reduced_mask_names


    def map_classes(self,lab,dict_map):
        
        # func works with tensors or (numpy) arrays
        if torch.is_tensor(lab):
            lab_new = lab.clone()
        else:
            lab_new = np.copy(lab)

        for c_old,c_new in dict_map.items():
            lab_new[lab == c_old] = c_new

        return lab_new


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
        lab = self.map_classes(lab,self.map) # in case if fewer classes are requested
        lab = np.unique(lab) # ensure that there are no doubles in label
        lab = self.one_hot(lab,self.new_class_encoding)
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
            if self.trafo_mode == 0:
                transform = A.Compose([
                    A.RandomCrop(height=small_side,width=small_side,always_apply=True),
                    A.Resize(height=256,width=256),
                    A.Defocus(radius=(1,5)),
                    A.RandomBrightnessContrast(p=2),
                    A.ColorJitter(),
                    A.HorizontalFlip(p=0.5),
                    A.GaussNoise(),
                    ToTensorV2()
                ])
            else:
                if mask_exist:
                    transform = A.Compose([
                        A.RandomResizedCrop(height=256,width=256,always_apply=True,scale=(0.01, 1.0),ratio=(1,1)),
                        A.Defocus(radius=(1,4)),
                        A.RandomBrightnessContrast(p=2),
                        A.HorizontalFlip(p=0.5),
                        A.GaussNoise(),
                        ToTensorV2()
                    ])
                else:
                    transform = A.Compose([
                        A.LongestMaxSize(max_size=256),
                        A.PadIfNeeded(min_height=256, min_width=256,p=1.0, border_mode=0),
                        A.Defocus(radius=(1,5)),
                        A.RandomBrightnessContrast(p=2),
                        A.HorizontalFlip(p=0.5),
                        A.GaussNoise(),
                        ToTensorV2()
                    ])
        else:
            if self.trafo_mode == 0:
                transform = A.Compose([
                    # A.RandomCrop(height=small_side,width=small_side,always_apply=True),
                    A.CenterCrop(height=small_side,width=small_side,always_apply=True),
                    A.Resize(height=256,width=256),
                    ToTensorV2()])
            else:
                transform = A.Compose([
                    A.LongestMaxSize(max_size=256),
                    A.PadIfNeeded(min_height=256, min_width=256, p=1.0, border_mode=0),
                    ToTensorV2()])

        transformed = transform(image=img,mask=mask) 
        mask = transformed["mask"] # HxW
        mask = mask.type(torch.float32)
        img = transformed["image"] # HxWxC -> CxHxW
        img = ((img.type(torch.float32)+1)/256.0).type(torch.float32) # TODO maybe normalize 
    
        # map to new rough or subset classes (identety map when turned of)
        mask = self.map_classes(mask,self.map)

        # get label from transformed mask -> only if mask exists
        if mask_exist:
            lab = self.get_label_from_mask(mask) # replace the one_hot function
            lab[0] = 1 # not important just for consistency

        # normalize img
        if self.normalization:
            img,mask=self.normalize(img,mask,mean=torch.tensor([0.485, 0.456, 0.406]), std=torch.tensor([0.229, 0.224, 0.225]))

        if self.return_img_idx==False:
            return img, lab, mask, mask_exist
        else:
            return img, lab, mask, mask_exist, idx

    
    def normalize(self,img, label, mean, std):
        if std is None:
            img-=mean[:,None, None]
        else:
            img-=mean[:,None, None]
            img/=std[:,None, None]
        return img,label

        

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





class Lizard_dataset(torch.utils.data.Dataset):
    def __init__(self, **kwargs):
        self.path= kwargs['path'] + 'lizard-challenge' # TODO change dataset structure
        self.mode=kwargs['mode']
        self.ccl_list=kwargs['classify_class_label_list']
        self.data_split=[4000,500,481]
        self.return_img_idx=False

        self.sub_imgs,self.sub_labels, self.sub_counts,self.num_data = self.load_data(kwargs['seed'])


        self.color_class_match={
            "0": ("black","background"),
            "1": ("lime", "neutrophil"),
            "2": ("white", "epithelial"),
            "3": ("cyan", "lymphocyte"),
            "4": ("blue", "plasma"),
            "5": ("red", "eosinophil"),
            "6": ("olive", "connective")
        }
        self.print_classify_class_info()
        # about the augmentation functions
        if kwargs['basic_aug_params'] is not None and self.mode=="train":
            self.basic_aug_fn=SpatialAugmenter(kwargs['basic_aug_params'])
        else:
            self.basic_aug_fn=None
        if  kwargs['color_aug_params'] is not None and self.mode=="train":
            self.color_aug_fn=color_augmentations(kwargs['color_aug_params'])
        else:
            self.color_aug_fn=None

    def __len__(self):
        return self.num_data

    def load_data(self,dataset_seed):
        if self.mode=="train":
            start=0
            end=self.data_split[0]
        elif self.mode=="val":
            start=self.data_split[0]
            end=self.data_split[0]+self.data_split[1]
        elif self.mode=="test":
            start=self.data_split[0]+self.data_split[1]
            end=self.data_split[0]+self.data_split[1]+self.data_split[2]
        else:
            raise NotImplementedError("The mode parameter can only be 'train', 'val' and 'test'.")


        #permulate the data
        np.random.seed(dataset_seed) #todo check if it is necessary as train.py has seed
        total_imgs_num=np.load(os.path.join(self.path,"images.npy")).shape[0]
        total_counts_num=self.read_csv().shape[0]
        split_sum=np.array(self.data_split).sum()
        assert split_sum==total_imgs_num and split_sum==total_counts_num, f"The sum of train, val, test data numbers {split_sum} shoud be equal to {total_imgs_num} and {total_counts_num}."
        seed_ind=np.random.permutation(total_imgs_num)

        #pick the sub_imgs, sub_labels
        sub_imgs=np.load(os.path.join(self.path,"images.npy"))[seed_ind][start:end,:,:,:]
        sub_labels=np.load(os.path.join(self.path,"labels.npy"))[seed_ind][start:end,:,:,:]
        sub_counts=self.read_csv()[seed_ind][start:end,:]
       

        return sub_imgs, sub_labels, sub_counts, end-start

    def __getitem__(self,idx):
        
       
            
        if self.mode=="train" or self.mode=="val":
            
            raw=self.sub_imgs[idx,:,:,:]  #shape (H,W,3) uint8 [0,255]
            sem_gt=self.sub_labels[idx,:,:,1:2].astype(np.uint8) #shape (H, W) uint16->unit8, {0,1,2,3,...,6} 
            if self.mode=="train":          
                raw,sem_gt=self.augment_sample(raw,sem_gt) 
                # raw [H,W,C]->[C,H,W], numpy.array->torch.tensor unit8->float32
                # sem_gt [H,W,C], numpy.array unit8 value~[0,6]
                #######################
                # aug_raw,aug_sem_gt=self.augment_sample(raw,sem_gt)
                # show_raw_sem_gt_and_aug(raw,sem_gt,aug_raw,aug_sem_gt) #check the augmentation images
                ################## 
            else:
                raw=transforms.ToTensor()(raw) # raw [H,W,C]->[C,H,W], numpy.array->torch.tensor unit8->float32
        else:

            raw=self.sub_imgs[idx,:,:,:]
            raw=transforms.ToTensor()(raw)
            sem_gt=self.sub_labels[idx,:,:,1:2].astype(np.uint8) # todo check why before we use self.sub_labels[idx,:,:,1:2]
            
        
        #TODO check
        
        # label=self.get_label(sem_gt) # or get label from counts.csv file
        label=self.get_label_from_counts(self.sub_counts[idx])
        
        if self.return_img_idx==False:
            return raw, label, sem_gt[:,:,0]
        else:
            return raw,label, sem_gt[:,:,0],idx


    def get_label(self, sem_gt):
        label=[]
        for i in self.ccl_list:
            if i in sem_gt:
                label.append(1)
            else:
                label.append(0)
        
        return np.array(label) # int64

    def get_label_from_counts(self,counts_array):
        counts=(counts_array>0.5).astype(int)
        return counts[self.ccl_list]


    def augment_sample(self,raw,sem_gt):
        if self.basic_aug_fn:
            raw,sem_gt=self.basic_aug_fn.forward_transform(raw,sem_gt) 
        # after the basic augmentation, dtype uint8, uint8,numpy.array, numpy.array value~(0-255) (0-6) [H,W,C]
         

        # Converts a PIL Image or numpy.ndarray (H x W x C) in the range [0, 255] to 
        # a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
        raw=transforms.ToTensor()(raw) # transfer the raw from numpy to tensor 
     
        if self.color_aug_fn:
            raw=self.color_aug_fn(raw)
        #raw [C, H, W] float32, torch.tensor value~(1.02--0.0)
        #sem_gt keep same [H,W, C] unit8 numpy.array value (0-6)

       
        return raw,sem_gt



    def get_class_weights_all(self):
        if len(self.ccl_list)==1:
            # this is binary classification case
            target_list=[]
            for i in range(self.sub_imgs.shape[0]):
                sem_gt=self.sub_labels[i,:,:,1:2].astype(np.uint8) #shape (H, W) uint16->unit8, {0,1,2,3,...,6} 
                if self.ccl_list[0] in sem_gt:
                    target_list.append(1)
                else:
                    target_list.append(0)
            values,counts=np.unique(np.array(target_list),return_counts=True) # order from samll to large, type=numpy array
            class_weights = 1./torch.tensor(list(counts), dtype=torch.float)
            class_weights_all = class_weights[target_list]
            return class_weights_all
        else:
            # multi-label classification
            sub_binarycounts=torch.from_numpy((self.sub_counts>0.5).astype(int)) # n_samples x 7 
            sub_binarycounts=sub_binarycounts[:,self.ccl_list] # n_samples x n_classes 
            class_probability=sub_binarycounts.sum(0, keepdims=True)/sub_binarycounts.shape[0] # 1 x n_classes
            sampling_weights=sub_binarycounts*class_probability + \
                        (torch.ones_like(sub_binarycounts)-sub_binarycounts)*(torch.ones_like(class_probability)-class_probability) # n_samples x n_classes 

            sampling_weights=1/sampling_weights.sum(1) # n_samples
            # print(sampling_weights.min(),sampling_weights.max()," min and max value of sampling weights")
            # print(sampling_weights.argmin(),sampling_weights.argmax(),"argmin amd argmax")  
            # print(sub_binarycounts[1],sub_binarycounts[2], "sub binary counts")
    
        return sampling_weights


    def print_classify_class_info(self):
        info="The nuclei classes to be classified: "
        for ccl in self.ccl_list:
            info= info + self.color_class_match[str(ccl)][1]+ ":" +self.color_class_match[str(ccl)][0] + " ; "
        print(info)


    def get_satistics(self):
        count_category=np.zeros((2**6,))
        patch_type_category=self.gen_patch_type_category(6)

        count_list=[]
        for i in range(self.sub_imgs.shape[0]):
            sem_gt=self.sub_labels[i,:,:,1:2].astype(np.uint8) #shape (H, W) uint16->unit8, {0,1,2,3,...,6} 
            tmp_list=[]
            for c in range(1,7):
                if c in sem_gt:
                    tmp_list.append(1)
                else:
                    tmp_list.append(0)
            
            idx=patch_type_category.index(tmp_list)
            count_category[idx]+=1
            count_list.append(np.array(tmp_list))  

        counts = np.stack(count_list,axis=0) # n_samples x classes 
        class_ratio=counts.sum(0)/self.num_data # classes
        
        
        number_case_list=[]
        for i in range(0,7):
            number_case_list.append((counts.sum(1)==i).sum()) #(7,)
        single_case_list=[]
        for i in range(0, 6):
            tmp_array=np.zeros(6)
            tmp_array[i]=1
            tmp_count=0
            for j in range(counts.shape[0]):
           
                if (counts[j,:]==tmp_array).all():
                    tmp_count+=1

            single_case_list.append(tmp_count) #(6,)

        assert sum(count_category)==self.num_data, f"The sum of data should be {self.num_data}"
        return class_ratio,number_case_list,single_case_list,patch_type_category,count_category
    

    def gen_patch_type_category(self,num_classes):
        # do not use self.num_classes; set num_classes=6
        patch_type_category={}
        catelist=[[0],[1]]
    
        for j in range(num_classes-1):
            new_list=[]
            for i in range(len(catelist)):

                a=copy.copy(catelist[i])
                a.append(0)
        
                b=copy.copy(catelist[i])
                b.append(1)
                new_list.append(a)
            
                new_list.append(b)
            
            catelist=copy.copy(new_list)
        
        # for i in range(len(catelist)):
        #     print(f"{i} : catelist {catelist[i]}")
        return catelist


    def read_csv(self):

        columns = defaultdict(list) # each value in each column is appended to a list
        csv_path=os.path.join(self.path,"counts.csv")
        with open(csv_path) as f:
            reader = csv.DictReader(f) # read rows into a dictionary format
            for row in reader: # read a row as {column1: value1, column2: value2,...}
                for (k,v) in row.items(): # go over each column name and value 
                    columns[k].append(int(v)) # append the value into the appropriate list
                                        # based on column name k

        c1=np.array(columns['neutrophil'])
        c2=np.array(columns['epithelial'])
        c3=np.array(columns['lymphocyte'])
        c4=np.array(columns['plasma'])
        c5=np.array(columns['eosinophil'])
        c6=np.array(columns['connective'])

        c0=np.ones_like(c1) # background array


        return np.stack([c0,c1,c2,c3,c4,c5,c6],axis=1)





class SpatialAugmenter(torch.nn.Module,):

    def __init__(self, params):
        '''
        params= {
            'fliplr': {'prob': 0.5},
            'flipud': {'prob': 0.5},
            'rotate90': {'start_factor': 1, 'end_factor': 4},
            'gammacontrast': {'gamma':(0.5,2.0)},
            'elastic': {'alpha': (0,40.0),'sigma':10 }
            }
        '''
        super(SpatialAugmenter, self).__init__()
        self.params = params
        trans_list=[]
        for key in self.params.keys():
            func=getattr(self,key)
            trans_list.append(func())  
        if trans_list:
            self.trans_fun= iaa.Sequential(trans_list,random_order=True)
        else:
            self.trans_fun=iaa.Identity()

        
    def forward_transform(self, raw, sem_gt=None):
        if sem_gt is not None:
            # this code makes sure that the same geometric augmentations are applied
            # to both the raw image and the label image
            sem_gt = SegmentationMapsOnImage(sem_gt, shape=raw.shape)
            
            raw,  sem_gt = self.trans_fun(image=raw, segmentation_maps= sem_gt)
            sem_gt =  sem_gt.get_arr()

            # some pytorch version have problems with negative indices introduced by e.g. flips
            # just copying fixes this
            sem_gt = sem_gt.copy()
            raw = raw.copy()
            return raw, sem_gt
        else:
            raw=self.transform_func(images=raw)
            raw = raw.copy()
            return raw, None


    def fliplr(self):
        prob=self.params['fliplr']['prob']
        return iaa.Fliplr(prob)


    def flipud(self):
        prob=self.params['flipud']['prob']
        return iaa.Flipud(prob)


    def rotate90(self):
        start=int(self.params['rotate90']['start_factor'])
        end=int(self.params['rotate90']['end_factor'])
        return iaa.Rot90((start,end),keep_size=False)


    def gammacontrast(self):
        gamma=self.params['gammacontrast']['gamma']
        assert isinstance(gamma, tuple), "gamma should be a tuple with two float value"
        return iaa.GammaContrast(gamma) #do scaling for the value of all channels with the same factor, not per channel a factor


    def elastic(self):
        alpha=self.params['elastic']['alpha']
        sigma=self.params['elastic']['sigma']
        return iaa.ElasticTransformation(alpha=alpha, sigma=sigma)

 
class GaussianNoise(torch.nn.Module):
    def __init__(self, sigma):
        super().__init__()
        self.sigma = sigma
        
    def forward(self, img):
        device = img.device
        noise = torch.randn(img.shape).to(device) * self.sigma
        return img + noise


def color_augmentations(size, s=0.5):
    # taken from https://github.com/sthalles/SimCLR/blob/master/data_aug/contrastive_learning_dataset.py
    """Return a set of data augmentation transformations as described in the SimCLR paper."""
    color_jitter = ColorJitter(0.8 * s, 0.0 * s, 0.8 * s, 0.2 * s) # brightness, contrast, saturation, hue
    HED_contrast = torch.nn.Sequential(
                Rgb2Hed(),
                LinearContrast(alpha=(0.65,1.35)),
                Hed2Rgb())
    data_transforms = torch.nn.Sequential(
        RandomApply([
            HED_contrast,
            color_jitter,
            GaussianNoise(0.005), 
            GaussianBlur(kernel_size=3, sigma=(0.1,0.1))], p=0.5),
        )
    return data_transforms







### old stuff 
'''
# The dataset contains 2913 images with a max image side of 500 px
# In order to construct a realistic experiment a fixed number of random crops was taken (imgname_idx)
# Of these fixed crops one can savely infer the class labels from the masks without adding information through the training process
img_names = sorted(os.listdir(self.path+'trainval_images'))
mask_names = sorted(os.listdir(self.path+'trainval_masks'))
names_og = self.read_list(self.path+'ImageSets/Segmentation/trainval.txt')
n_crops = int(len(img_names)/len(names_og))

# get a subset of images of the full dataset
idx0_train = 0
idx0_val = len(img_names)-n_crops*(self.len_val+self.len_test)
idx0_test = len(img_names)-n_crops*self.len_test

# define subset classes -> move to utils
sub_classes_names = ['bird','car','cat','chair','dog','people']
classes_rough_map = {'bg':[0],'plane':[1],'animal':[3,8,10,12,13,17],'ground_vehicle':[2,4,6,7,14,19],'computer':[20],'sofa_chair_table':[18,9,11],'person':[15]}
sub_classes = get_key(sub_classes_names,classes) 
'''




'''
old getimage on pascal
# setup transformation pipeline


# apply all trafos to image


if mode == 'subset':
    # delete classes and infer class label
    class_label = sum(torch.Tensor(self.sub_classes) == c for c in torch.unique(mask))
    class_label = class_label.type(torch.float32)

    new_mask = 0*mask
    for i in range(len(self.sub_classes)):
        new_mask += i*(mask == self.sub_classes[i]) # only for 7 class setup
    mask = new_mask.type(torch.float32)

if mode == 'rough_classes':
    # infer new classes
    n_classes = len(self.rough_classes)
    map = list(self.rough_classes.values())
    class_label = torch.zeros(n_classes).type(torch.float32)
    img_classes = torch.unique(mask).detach().cpu().numpy()
    for i in range(n_classes):
        class_label[i] = ([item for item in map[i] if item in img_classes] != [])
    class_label = class_label.type(torch.float32)

    # make new mask
    new_mask = 0*mask
    for i in range(len(class_label)):
        if class_label[i]:
            for og_class in map[i]:
                new_mask += (i)*(mask == og_class) # only for 7 classes
    mask = new_mask.type(torch.float32)
'''