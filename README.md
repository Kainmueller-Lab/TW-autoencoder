# TW-autoencode - Tied-weights AutoEncoder

This repository contains the source code of unrolled LRP model and baselines from the paper, [**Model Guidance via Explanations Turns Image
Classifiers into Segmentation Models**](https://arxiv.org/pdf/2407.03009).


## Updates

**Apr. 2024** -- Start to clean the code.

**July. 2024** -- Publish the first version of code.

## Datasets

- For Pascal VOC, please download the original training images from the [official PASCAL site](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar): `VOCtrainval_11-May-2012.tar`. Extract the folder `JPEGImages` and `SegmentationClassAug` into the corresponding `dataset/pascal` folder.

- The data folder structure should look like the following, and the data_path argument should be `--data_path /path/to/data_parent_folder`.
```
data_parent_folder
└───VOC2012
    │   train.txt
    │   val.txt
    └───Annotations
    └───ImageSets
    └───JPEGImages
    └───SegmentationObject
```

## Training Supervised and Semi-supervised Models

- Supervised training for UNets in 4 different cases (20, 100, 500, 1464 pixel-labeled data)

```
python train.py --model std_unet --batch_size 10 --pretrain_weight_name ./snapshot/resnet50_10_pre_train_21 --encoder resnet50 --seed 42 --epochs 15000 --num_labels 20 --uniform_masks 1.0 --save_folder std_unet_resnet50_lab20_lr1e-5_s42
```

```
python train.py --model std_unet --batch_size 10 --pretrain_weight_name ./snapshot/resnet50_10_pre_train_21 --encoder resnet50 --seed 42 --epochs 3000 --num_labels 100 --uniform_masks 1.0 --save_folder std_unet_resnet50_lab100_lr1e-5_s42
```

```
python train.py --model std_unet --batch_size 10 --pretrain_weight_name ./snapshot/resnet50_10_pre_train_21 --encoder resnet50 --seed 42 --epochs 600 --num_labels 500 --uniform_masks 0.5 --save_folder std_unet_resnet50_lab500_lr1e-5_s42
```

```
python train.py --model std_unet --batch_size 10 --pretrain_weight_name ./snapshot/resnet50_10_pre_train_21 --encoder resnet50 --seed 42 --epochs 200 --save_folder std_unet_resnet50_lab1464_lr1e-5_s42
```
- Semi-supervised training for multi-task UNets and Unrolled LRP models in 4 different cases (20, 100, 500, 1464 pixel-labeled data). For multi-task UNets, change  `--model unrolled_lrp` to `--model mt_unet`.


```
python train.py --model unrolled_lrp --semisup_dataset --add_classification --iterative_gradients --batch_size 10 --pretrain_weight_name ./snapshot/resnet50_10_pre_train_21 --encoder resnet50 --seed 42 --epochs 100 --num_labels 20 --uniform_masks 1.0 --save_folder lrp0_resnet50_lab20_lr1e-5_s42
```

```
python train.py --model unrolled_lrp --semisup_dataset --add_classification --iterative_gradients --batch_size 10 --pretrain_weight_name ./snapshot/resnet50_10_pre_train_21 --encoder resnet50 --seed 42 --epochs 100 --num_labels 100 --uniform_masks 1.0 --save_folder lrp0_resnet50_lab100_lr1e-5_s42 
```

```
python train.py --model unrolled_lrp --semisup_dataset --add_classification --iterative_gradients --batch_size 10 --pretrain_weight_name ./snapshot/resnet50_10_pre_train_21 --encoder resnet50 --seed 42 --epochs 100 --num_labels 500 --uniform_masks 0.5 --save_folder lrp0_resnet50_lab500_lr1e-5_s42
```

```
python train.py --model unrolled_lrp --semisup_dataset --add_classification --iterative_gradients --batch_size 10 --pretrain_weight_name ./snapshot/resnet50_10_pre_train_21 --encoder resnet50 --seed 42 --epochs 100 --save_folder lrp0_resnet50_lab1464_lr1e-5_s42
```

Note: The way for counting epochs differs between the semisupervised and supervised datasets. Therefore, you need to set a higher value for the epochs argument when training the UNet.


## Citation

Yu, Xiaoyan, et al. "Model Guidance via Explanations Turns Image Classifiers into Segmentation Models." World Conference on Explainable Artificial Intelligence. Cham: Springer Nature Switzerland, 2024.


## Contact
If you have any questions, please contact xiaoyany1101@gmail.com.