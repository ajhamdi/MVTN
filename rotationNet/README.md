# Pytorch RotationNet

This is a pytorch implementation of RotationNet.

Asako Kanezaki, Yasuyuki Matsushita and Yoshifumi Nishida.
**RotationNet: Joint Object Categorization and Pose Estimation Using Multiviews from Unsupervised Viewpoints.** 
*CVPR*, accepted, 2018.
([pdf](https://arxiv.org/abs/1603.06208))
([project](https://kanezaki.github.io/rotationnet/))

We used caffe for the CVPR submission.
Please see [rotationnet](https://github.com/kanezaki/rotationnet) repository for more details including how to reproduce the results in our paper.

## Training/testing ModelNet dataset

### 1. Download multi-view images
#### 1-1. Download multi-view images generated in [Su et al. 2015]
    $ bash get_modelnet_png.sh  
[Su et al. 2015] H. Su, S. Maji, E. Kalogerakis, E. Learned-Miller. Multi-view Convolutional Neural Networks for 3D Shape Recognition. ICCV2015.  
This is a subset of ModelNet40.
#### 1-2. Download our multi-view images 
    $ wget https://data.airc.aist.go.jp/kanezaki.asako/data/modelnet40v2png_ori4.tar; tar xvf modelnet40v2png_ori4.tar  
Our BEST results are reported on this dataset.

### 2. Prepare dataset directories for training
    $ bash link_images.sh ./modelnet40v1png ./ModelNet40v1 1  
    $ bash link_images.sh ./modelnet40v2png ./ModelNet40_20 2  
Or  

    $ bash link_images.sh ./modelnet40v2png_ori4 ./ModelNet40_20  

### 3. Train your own RotationNet models
#### 3-1. Case (2): Train the model w/o upright orientation (RECOMMENDED)
    $ python train_rotationnet.py --pretrained -a alexnet -b 400 --lr 0.01 --epochs 1500 ./ModelNet40_20 | tee log_ModelNet40_20_rotationnet.txt
#### 3-2. Case (1): Train the model with upright orientation
    $ python train_rotationnet.py --case 1 --pretrained -a alexnet -b 240 --lr 0.01 --epochs 1500 ./ModelNet40v1 | tee log_ModelNet40v1_rotationnet.txt 

## Training/testing MIRO dataset

### 1. Download MIRO dataset (414MB)
    $ wget https://data.airc.aist.go.jp/kanezaki.asako/data/MIRO.zip  
    $ unzip MIRO.zip 

### 2. Prepare dataset directories for training
    $ bash link_images_MIRO.sh ./MIRO ./data_MIRO

### 3. Train your own RotationNet models
#### 3-1. Case (3): Train the model w/ upright orientation
    $ python train_rotationnet.py --case 3 --pretrained -a alexnet -b 480 --lr 0.01 --epochs 1500 ./data_MIRO | tee log_MIRO_160_rotationnet.txt
