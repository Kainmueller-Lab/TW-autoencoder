# TW-autoencode - Tied-weights AutoEncoder

This repository contains the source code of unrolled LRP model and baselines from the paper, **Model Guidance via Explanations Turns Image
Classifiers into Segmentation Models**.

- TODO add link for the paper.


## Updates

**Apr. 2024** -- Clean the code.

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
qsub train.sh train.py --model std_unet --data_path  /fast/AG_Kainmueller/xyu/ --batch_size 10 --pretrain_weight_name ./snapshot/resnet50_10_pre_train_21 --encoder resnet50 --seed 40 --epochs 15000 --num_labels 20 --uniform_masks 1.0 --save_folder std_unet_resnet50_lab20_lr1e-5_s40
```

```
qsub train.sh train.py --model std_unet --data_path  /fast/AG_Kainmueller/xyu/ --batch_size 10 --pretrain_weight_name ./snapshot/resnet50_10_pre_train_21 --encoder resnet50 --seed 40 --epochs 3000 --num_labels 100 --uniform_masks 1.0 --save_folder std_unet_resnet50_lab100_lr1e-5_s40
```

```
qsub train.sh train.py --model std_unet --data_path  /fast/AG_Kainmueller/xyu/ --batch_size 10 --pretrain_weight_name ./snapshot/resnet50_10_pre_train_21 --encoder resnet50 --seed 40 --epochs 600 --num_labels 500 --uniform_masks 0.5 --save_folder std_unet_resnet50_lab500_lr1e-5_s40
```

```
qsub train.sh train.py --model std_unet --data_path  /fast/AG_Kainmueller/xyu/ --batch_size 10 --pretrain_weight_name ./snapshot/resnet50_10_pre_train_21 --encoder resnet50 --seed 40 --epochs 200 --save_folder std_unet_resnet50_lab1464_lr1e-5_s40
```
- Semi-supervised training for multi-task UNets and Unrolled LRP models in 4 different cases (20, 100, 500, 1464 pixel-labeled data). For multi-task UNets, change  `--model unrolled_lrp` to `--model mt_unet`.


```
qsub train.sh train.py --model unrolled_lrp --semisup_dataset --data_path /fast/AG_Kainmueller/xyu/ --batch_size 10 --pretrain_weight_name ./snapshot/resnet50_10_pre_train_21 --encoder resnet50 --seed 40 --epochs 100 --iterative_gradients --add_classification --num_labels 20 --uniform_masks 1.0 --save_folder lrp0_resnet50_lab20_lr1e-5_s40 
```

```
qsub train.sh train.py --model unrolled_lrp --semisup_dataset --data_path /fast/AG_Kainmueller/xyu/ --batch_size 10 --pretrain_weight_name ./snapshot/resnet50_10_pre_train_21 --encoder resnet50 --seed 40 --epochs 100 --iterative_gradients --add_classification --num_labels 100 --uniform_masks 1.0 --save_folder lrp0_resnet50_lab100_lr1e-5_s40 
```

```
qsub train.sh train.py --model unrolled_lrp --semisup_dataset --data_path /fast/AG_Kainmueller/xyu/ --batch_size 10 --pretrain_weight_name ./snapshot/resnet50_10_pre_train_21 --encoder resnet50 --seed 40 --epochs 100 --iterative_gradients --add_classification --num_labels 500 --uniform_masks 0.5 --save_folder lrp0_resnet50_lab500_lr1e-5_s40 
```

```
qsub train.sh train.py --model unrolled_lrp --semisup_dataset --data_path /fast/AG_Kainmueller/xyu/ --batch_size 10 --pretrain_weight_name ./snapshot/resnet50_10_pre_train_21 --encoder resnet50 --seed 40 --epochs 100 --iterative_gradients --add_classification --save_folder lrp0_resnet50_lab1464_lr1e-5_s40 
```


- TODO rewrite above comments which is used for running on the max cluster

example
```
python train.py --model std_unet --batch_size 10 --pretrain_weight_name ./snapshot/resnet50_10_pre_train_21 --encoder resnet50 --seed 42 --epochs 15000 --num_labels 20 --uniform_masks 1.0 --save_folder std_unet_resnet50_lab20_lr1e-5_s42
```


```
python train.py --model unrolled_lrp --semisup_dataset --add_classification --iterative_gradients --batch_size 10 --pretrain_weight_name ./snapshot/resnet50_10_pre_train_21 --encoder resnet50 --seed 42 --epochs 15000 --num_labels 20 --uniform_masks 1.0 --save_folder lrp0_resnet50_lab20_lr1e-5_s42
```

- TODO write conda environment for running the code.


## Citation

- TODO


## Contact
If you have any questions, please contact xiaoyany1101@gmail.com.