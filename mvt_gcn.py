import torch
import torch.nn as nn
import sys
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.autograd import Variable
import warnings
# warnings.filterwarnings("error")
# torch.multiprocessing.set_start_method('spawn')
from util import *
from ops import *
import numpy as np
import random
import torch.optim as optim
import shutil
import json
from viewGCN.tools.Trainer_mvt import ModelNetTrainer_mvt
from viewGCN.tools.ImgDataset import MultiviewImgDataset, SingleImgDataset
from viewGCN.model.view_gcn import view_GCN, SVCNN

# 3D transformations functions


import torchvision.transforms as transforms

import argparse
import time
import os

# from models.resnet import *
# from models.mvcnn import *
# from models.pointnet import *


# from logger import Logger
from torch.utils.tensorboard import SummaryWriter
from custom_dataset import MultiViewDataSet, ThreeMultiViewDataSet, collate_fn, ShapeNetCore, ScanObjectNN  # , ModelNet40

# MVCNN = 'mvcnn'
# RESNET = 'resnet'
# NETWORKS = [RESNET,MVCNN]








parser = argparse.ArgumentParser(description='ViewGCN-MVT')
parser.add_argument('--depth', choices=[18, 34, 50, 101, 152], type=int,  default=18, help='resnet depth (default: resnet18)')
parser.add_argument('--gpu', type=int,
                     default=0, help='GPU number ')
# parser.add_argument('--mvnetwork', '-m',  default=RESNET, choices=NETWORKS,
#                     help='pretrained mvnetwork: ' + ' | '.join(NETWORKS) + ' (default: {})'.format(RESNET))
parser.add_argument('--epochs', default=100, type=int,  help='number of total epochs to run (default: 100)')
parser.add_argument('-b', '--batch_size', default=16, type=int,
                     help='mini-batch size (default: 4)')

parser.add_argument('--image_data',required=True,  help='path to 2D dataset')
parser.add_argument('--mesh_data', required=True,  help='path to 3D dataset')
parser.add_argument('--exp_set', type=str, default='00', help='pick ')
parser.add_argument('--exp_id', type=str, default='random', help='pick ')
parser.add_argument('--nb_views', default=12, type=int, 
                    help='number of views in MV CNN')
parser.add_argument('--nb_points', default=2048, type=int,help='number of points in the 3d dataeset sampled from the meshes ')
parser.add_argument('--image_size', default=224, type=int, 
                    help='the size of the images rendered by the differntibe renderer')
parser.add_argument('--canonical_elevation', default=30.0, type=float,
                     help='if selection_type== canoncal , the elevation of the view points is givene by this angle')
parser.add_argument('--canonical_distance', default=2.2, type=float,
                     help='the distnace of the view points from the center if the object  ')
parser.add_argument('--input_view_noise', default=0.0, type=float,
                    help='the variance of the gaussian noise (before normalization with parametre range) added to the azim,elev,dist inputs to the MVT ... this option is valid only if `learned_offset` or `learned_direct` options are sleected   ')
parser.add_argument('--selection_type', '-s',  default="canonical", choices=["canonical", "random", "learned_offset", "learned_direct", "spherical", "learned_spherical", "learned_random", "learned_transfer"],
                    help='the selection type of views ')
parser.add_argument('--plot_freq', default=2, type=int, 
                    help='the frequqency of plotting the renderings and camera positions')
parser.add_argument('--GCN_dir', '-gcnd',  default="viewGCN",help='the view-GCN folder')
parser.add_argument('--renderings_dir', '-rd',  default="renderings",help='the destinatiojn for the renderings ')
parser.add_argument('--results_dir', '-rsd',  default="results",help='the destinatiojn for the results ')
parser.add_argument('--logs_dir', '-lsd',  default="logs",help='the destinatiojn for the tensorboard logs ')
parser.add_argument('--cameras_dir', '-c', 
                    default="cameras", help='the destination for the 3D plots of the cameras ')
parser.add_argument('--vs_learning_rate', default=0.0001, type=float,
                    help='initial learning rate for view selector (default: 0.00001)')
parser.add_argument('--pn_learning_rate', default=0.0001, type=float,
                    help='initial learning rate for view selector (default: 0.00005)')
parser.add_argument('--simplified_mesh', dest='simplified_mesh',
                    action='store_true', help='use simplified meshes in learning .. not the full meshes  ')
parser.add_argument('--view_reg', default=0.0, type=float,
                    help='use regulizer to the learned view selector so they can be apart ...ONLY when `selection_type` == learned_direct   (default: 0.0)')
parser.add_argument('--shape_extractor', '-pnet',  default="PointNet", choices=["PointNet", "DGCNN", "PointNetPP", "MVCNN"],
                    help='pretrained point cloud mvnetwork to get coarse featrures ')
parser.add_argument('--features_type', '-ftpe',  default="post_max", choices=["logits", "post_max", "transform_matrix",
                                                                              "pre_linear", "logits_trans", "post_max_trans", "pre_linear_trans"],
                    help='the type of the features extracted from the feature extractor ( early , middle , late) ')
parser.add_argument('--transform_distance', dest='transform_distance',
                    action='store_true', help='transform the distance to the object as well  ')
parser.add_argument('--vs_weight_decay', default=0.01, type=float,
                    help='weight decay for MVT ... default 0.01')
parser.add_argument('--clip_grads', dest='clip_grads',
                    action='store_true', help='clip the gradients of the MVT with L2= `clip_grads_value` ')
parser.add_argument('--vs_clip_grads_value', default=3, type=float,help='the clip value for L2 of gradeients of the MVT ')
parser.add_argument('--fe_clip_grads_value', default=3, type=float,
                    help='the clip value for L2 of gradeients of the Shape feature extractor .. eg.e PointENt ')


parser.add_argument('--momentum', default=0.9, type=float, 
                    help='momentum (default: 0.9)')
parser.add_argument('--lr-decay-freq', default=30, type=float,
                     help='learning rate decay (default: 30)')
parser.add_argument('--lr-decay', default=0.1, type=float,
                     help='learning rate decay (default: 0.1)')
parser.add_argument('--print_freq', '-p', default=30, type=int,
                     help='print frequency (default: 10)')
parser.add_argument('-r', '--resume', dest='resume',
                    action='store_true', help='continue training from the `setup[weights_file] checkpoint ')
parser.add_argument('-rr', '--resume_first', dest='resume_first',
                    action='store_true', help='continue training from the `setup[weights_file] checkpoint ')
parser.add_argument('-rrr', '--resume_second', dest='resume_second',
                    action='store_true', help='continue training from the `setup[weights_file] checkpoint ')
parser.add_argument('--pretrained', dest='pretrained', action='store_true', help='use pre-trained mvnetwork')
parser.add_argument('--save_all', dest='save_all',
                    action='store_true', help='save save the checkpoint and results at every epoch.... default saves only best test accuracy epoch')
parser.add_argument('--test_only', dest='test_only',
                    action='store_true', help='do only testing once ... no training ')
parser.add_argument('--train', dest='train',
                    action='store_true', help='do only training  once ... no testing ')
parser.add_argument('--test_only_retr', dest='test_only_retr',
                    action='store_true', help='do only testing once ... no training (output the retrival score)')
parser.add_argument('--LFDA_dimension', dest='LFDA_dimension',
                    default=64, type=int, help='dimension for LFDA projection (0 = no projection')
parser.add_argument('--LFDA_layer', dest='LFDA_layer',
                    default=0, type=int, help='layer for LFDA projection')
parser.add_argument('--return_points_sampled', dest='return_points_sampled',
                    action='store_true', help='reuturn 3d point clouds from the data loader sampled from hte mesh ')
parser.add_argument('--return_points_saved', dest='return_points_saved',
                    action='store_true', help='reuturn 3d point clouds from the data loader saved under `filePOINTS.pkl` ')
# parser.add_argument('--return_extracted_features', dest='return_extracted_features',
#                     action='store_true', help='return pre extracted features `*_PFeatures.pt` for each 3d model from the dataloader ')
parser.add_argument('--rotated_test', dest='rotated_test',
                    action='store_true', help=' test on rotation noise on the meshes from ModelNet40 to make it realistic  ')
parser.add_argument('--rotated_train', dest='rotated_train',
                    action='store_true', help=' train on rotation noise on the meshes from ModelNet40 to make it realistic  ')
# parser.add_argument('--noisy_dataset', '-ndset',  default="no", choices=["no", "low", "medium", "high"],
#                     help='the amount of rotation noise on the meshes from ModelNet40 ')
parser.add_argument('--log_metrics', dest='log_metrics',
                    action='store_true', help='logs loss and acuracy and other metrics in `logs_dir` for tensorboard ')
# parser.add_argument('--random_lighting', dest='random_lighting',
#                     action='store_true', help='apply random light direction on the rendered images .. otherwise default (0, 1.0, 0) ')
parser.add_argument('--light_direction', '-ldrct',  default="random", choices=["fixed", "random", "relative"],
                    help='apply fixed light from top or  random light direction on the rendered images or random .. otherwise default (0, 1.0, 0)')
parser.add_argument('--screatch_feature_extractor', dest='screatch_feature_extractor',
                    action='store_true', help='start training the feature extractor from scracth ...applies ONLY if `is_learning_points` == True ')


parser.add_argument("---name", type=str, help="Name of the experiment", default="view-gcn")
parser.add_argument("--second_stage_bs", type=int, help="Batch size for the second stage", default=20)# it will be *12 images in each batch for mvcnn
parser.add_argument("--second_stage_epochs", type=int,
                    help="number of epochs for the second stage", default=25)
parser.add_argument("--first_stage_bs", type=int,help="Batch size for the first stage", default=400)
parser.add_argument("--first_stage_epochs", type=int,
                    help="number of epochs for the first stage", default=30)
parser.add_argument("--num_models", type=int, help="number of models per class", default=0)
parser.add_argument("--learning_rate", type=float,help="learning rate second stage", default=1e-3)
parser.add_argument("--learning_rate_first", type=float,
                    help="learning rate for first stage", default=1e-4)

parser.add_argument("--weight_decay", type=float, help="weight decay", default=0.001)
# parser.add_argument("--cnn_name", "--cnn_name", type=str, help="cnn model name", default="resnet18")
parser.add_argument("--train_path", type=str, default="data/modelnet40v2png_ori4/*/train")
parser.add_argument("--val_path", type=str, default="data/modelnet40v2png_ori4/*/test")
parser.add_argument("--phase", default="all", choices=["all", "first", "second", "third"],
                    help='what stage of train/test of the VIEW-GCN ( MVCNN, or the gcn or the MVT)')
parser.add_argument('--resume_mvtn', dest='resume_mvtn',
                    action='store_true', help='use a pretrained MVTN and freez during this training  ')
# parser.add_argument('--normalize_properly', dest='normalize_properly',
                    # action='store_true', help='use true means and std of the training to normalize the rendered images  ')
parser.add_argument('--cull_backfaces', dest='cull_backfaces',
                    action='store_true', help='cull back_faces ( remove them from the image) ')
parser.add_argument('--cleaned_mesh', dest='cleaned_mesh',
                    action='store_true', help='use cleaned meshes using reversion of light direction for faulted meshes')

## point cloud rnedienring
parser.add_argument('--pc_rendering', dest='pc_rendering',
                    action='store_true', help='use point cloud renderer instead of mesh renderer  ')
parser.add_argument('--points_radius', default=0.006, type=float,
                    help='the size of the rendered points if `pc_rendering` is True  ')
parser.add_argument('--points_per_pixel',  default=10, type=int,
                    help='max number of points in every rendered pixel if `pc_rendering` is True ')
parser.add_argument('--dset_variant', '-dsetp',  default="obj_only",
                    choices=["obj_only", "with_bg", "hardest"])
# parser.add_argument('--nb_points', default=2048, type=int,
# help='number of points in the 3d dataeset sampled from the meshes ')
parser.add_argument('--object_color',  default="white", choices=["white", "random", "black", "red", "green", "blue", "learned", "custom"],
                    help='the selection type of views ')
parser.add_argument('--background_color', '-bgc',  default="white", choices=["white", "random", "black", "red", "green", "blue", "learned", "custom"],
                    help='the color of the background of the rendered images')
parser.add_argument('--dset_norm', '-dstn',  default="2", choices=["inf", "2", "1", "fro", "no"],
                    help='the L P normlization tyoe of the 3D dataset')
parser.add_argument('--initial_angle', default=-90.0, type=float,
                    help='the inital tilt angle of the 3D object as loaded from the ModelNet40 dataset  ')
parser.add_argument('--augment_training', dest='augment_training',
                    action='store_true', help='augment the training of the CNN by scaling , rotation , translation , etc ')
parser.add_argument('--crop_ratio', default=0.3, type=float,
                    help='the crop ratio of the images when `augment_training` == True  ')
parser.add_argument('--ignore_normalize', dest='ignore_normalize',
                    action='store_true', help='ignore any normalization performed on the image (mean,std) before passing to the network')
parser.add_argument('--occlusion_robustness_mode', dest='occlusion_robustness_mode',
                    action='store_true', help='use the point cloud rendering to test robustness to occlusion of chopping point cloud ')
parser.add_argument('--rotation_robustness_mode', dest='rotation_robustness_mode',
                    action='store_true', help='use the point cloud rendering to test robustness to rotation at test time')
parser.add_argument('--max_degs', default=180.0, type=float,
                    help='the maximum allowed Z rotation degrees on the meshes ')
parser.add_argument('--repeat_exp', '-rpx', default=3, type=int,
                    help='the number of repeated exps for each setup ( due to randomness)')
args = parser.parse_args()
setup = vars(args)
initialize_setup_gcn(setup)

print('Loading data')

# a function to preprocess pytorch3d Mesh onject

# device = torch.device("cuda:{}".format(str(setup["gpu"])) if torch.cuda.is_available() else "cpu")
# print('Running on ' + str(device))


# mvnetwork.cuda()
cudnn.benchmark = True



# Loss and Optimizer
lr = setup["learning_rate"]
n_epochs = setup["epochs"]
view_selector = ViewSelector(setup["nb_views"], selection_type=setup["selection_type"],
                             canonical_elevation=setup["canonical_elevation"],canonical_distance= setup["canonical_distance"],
                             shape_features_size=setup["features_size"], transform_distance=setup["transform_distance"], input_view_noise=setup["input_view_noise"], light_direction=setup["light_direction"]).cuda()
feature_extractor = FeatureExtracter(setup).cuda()
print(setup)
criterion = nn.CrossEntropyLoss()
views_criterion = nn.CosineSimilarity()
models_bag = {
    # "mvnetwork": mvnetwork,"optimizer": optimizer,
    "view_selector": view_selector,"feature_extractor": feature_extractor
    }

if setup["resume_mvtn"]:
    load_mvtn(setup, models_bag, setup["weights_file2"])
    setup["vs_learning_rate"] = 0.0
    setup["pn_learning_rate"] = 0.0

# optimizer = torch.optim.AdamW(    mvnetwork.parameters(), lr=lr, weight_decay=setup["weight_decay"])
if setup["is_learning_views"]:
    vs_optimizer = torch.optim.AdamW(view_selector.parameters(), lr=setup["vs_learning_rate"], weight_decay=setup["vs_weight_decay"])
else : 
    vs_optimizer = None
if setup["is_learning_points"]:
    fe_optimizer = torch.optim.AdamW(feature_extractor.parameters(), lr=setup["pn_learning_rate"])
else:
    fe_optimizer = None
models_bag["vs_optimizer"] = vs_optimizer
models_bag["fe_optimizer"] = fe_optimizer



def view_gcn_exp(setup, models_bag):
    seed_torch()
    if "modelnet" in setup["mesh_data"].lower():
        dset_train = ShapeNetCore(setup["mesh_data"], ("train",), setup["nb_points"], load_textures=False,
                                  dset_norm=setup["dset_norm"], simplified_mesh=setup["simplified_mesh"])
        dset_val = ShapeNetCore(setup["mesh_data"], ("test",), setup["nb_points"], load_textures=False,
                                dset_norm=setup["dset_norm"], simplified_mesh=setup["simplified_mesh"])
        classes = dset_train.classes


    elif "shapenetcore" in setup["mesh_data"].lower():
        dset_train = ShapeNetCore(setup["mesh_data"], ("train",), setup["nb_points"], load_textures=False,
                                dset_norm=setup["dset_norm"], simplified_mesh=setup["simplified_mesh"])
        dset_val = ShapeNetCore(setup["mesh_data"], ("test",), setup["nb_points"], load_textures=False,
                                dset_norm=setup["dset_norm"], simplified_mesh=setup["simplified_mesh"])
        # dset_train, dset_val = torch.utils.data.random_split(shapenet, [int(.8*len(shapenet)), int(np.ceil(0.2*len(shapenet)))])  #, generator=torch.Generator().manual_seed(42))   ## to reprodebel results
        classes = dset_val.classes
    elif "scanobjectnn" in setup["mesh_data"].lower():
        dset_train = ScanObjectNN(setup["mesh_data"], 'train',  setup["nb_points"],
                                variant=setup["dset_variant"], dset_norm=setup["dset_norm"])
        dset_val = ScanObjectNN(
            setup["mesh_data"], 'test',  setup["nb_points"], variant=setup["dset_variant"], dset_norm=setup["dset_norm"])
        classes = dset_train.classes

    train_loader = DataLoader(dset_train, batch_size=setup["batch_size"],
                            shuffle=True, num_workers=6, collate_fn=collate_fn, drop_last=True)

    val_loader = DataLoader(dset_val, batch_size=int(setup["batch_size"]/2),
                            shuffle=False, num_workers=6, collate_fn=collate_fn)

    print("classes nb:", len(classes), "number of train models: ", len(
        dset_train), "number of test models: ", len(dset_val), classes)



    # STAGE 1
    models_bag["mvnetwork"] = SVCNN(setup["exp_id"], nclasses=len(classes),
                 pretraining=setup["pretrained"], cnn_name=setup["cnn_name"])
    models_bag["optimizer"] = optim.SGD(models_bag["mvnetwork"].parameters(), lr=setup["learning_rate_first"],
                          weight_decay=setup["weight_decay"], momentum=0.9)
    # models_bag["optimizer"] = torch.optim.AdamW(models_bag["mvnetwork"].parameters(), lr=setup["learning_rate_first"],
    #                       weight_decay=setup["weight_decay"])
    n_models_train = setup["num_models"]*setup["nb_views"]


    models_bag["mvnetwork"].train()
    models_bag["view_selector"].train()
    models_bag["feature_extractor"].train()
    # train_dataset = SingleImgDataset(setup["train_path"], scale_aug=False,
    #                                  rot_aug=False, num_models=n_models_train, num_views=setup["nb_views"])
    # train_loader = torch.utils.data.DataLoader(
    #     train_dataset, batch_size=setup["first_stage_bs"], shuffle=True, num_workers=4)
    # val_dataset = SingleImgDataset(
    #     setup["val_path"], scale_aug=False, rot_aug=False, test_mode=True)
    # val_loader = torch.utils.data.DataLoader(
    #     val_dataset, batch_size=setup["first_stage_bs"], shuffle=False, num_workers=4)
    # print('num_train_files: '+str(len(train_dataset.filepaths)))
    # print('num_val_files: '+str(len(val_dataset.filepaths)))
    trainer = ModelNetTrainer_mvt(models_bag, train_loader, val_loader, dset_val, nn.CrossEntropyLoss(
    ), 'svcnn', setup["checkpoint_dir1"], num_views=1, setup=setup, classes=classes)
    # trainer.visualize_views("test", [5,55,150], torch.cuda.current_device())

    if setup["resume_first"]:
        trainer.model.load(trainer.weights_dir,)
    if setup["phase"] == "all" or setup["phase"] == "first":
        if not setup["test_only"]:
            trainer.train(setup["first_stage_epochs"])
        else:
            trainer.visualize_views("test", [55, 66,77])
            trainer.update_validation_accuracy(1)

    
    
    # # # STAGE 2
    models_bag["mvnetwork"] = view_GCN(setup["exp_id"], models_bag["mvnetwork"], nclasses=len(classes),
                   cnn_name=setup["cnn_name"], num_views=setup["nb_views"])
    models_bag["optimizer"] = optim.SGD(models_bag["mvnetwork"].parameters(), lr=setup["learning_rate"],
                          weight_decay=setup["weight_decay"], momentum=0.9)
    # models_bag["optimizer"] = torch.optim.AdamW(models_bag["mvnetwork"].parameters(), lr=setup["learning_rate_first"],
    #                                             weight_decay=setup["weight_decay"])
    # train_dataset = MultiviewImgDataset(setup["train_path"], scale_aug=False, rot_aug=False,
    #                                     num_models=n_models_train, num_views=setup["nb_views"], test_mode=True)
    # shuffle needs to be false! it's done within the trainer
    # train_loader = torch.utils.data.DataLoader(
    #     train_dataset, batch_size=setup["second_stage_bs"], shuffle=False, num_workers=4)
    # val_dataset = MultiviewImgDataset(
    #     setup["val_path"], scale_aug=False, rot_aug=False, num_views=setup["nb_views"], test_mode=True)
    # val_loader = torch.utils.data.DataLoader(
    #     val_dataset, batch_size=setup["second_stage_bs"], shuffle=False, num_workers=4)
    # print('num_train_files: '+str(len(train_dataset.filepaths)))
    # print('num_val_files: '+str(len(val_dataset.filepaths)))
    trainer = ModelNetTrainer_mvt(models_bag, train_loader, val_loader, dset_val,
                                  nn.CrossEntropyLoss(), 'view-gcn', setup["checkpoint_dir2"], num_views=setup["nb_views"], setup=setup, classes=classes)
    #use trained_view_gcn
    #gcn.load_state_dict(torch.load('trained_view_gcn.pth'))
    #trainer.update_validation_accuracy(1)
    if setup["resume_second"] :
        trainer.model.load(trainer.weights_dir,)
        if setup["is_learning_views"]:
            load_mvt(setup, models_bag, setup["weights_file2"])
    if setup["phase"] == "all" or setup["phase"] == "second":
        if setup["train"]:
            trainer.train(setup["second_stage_epochs"])
        if setup["test_only"]:
            trainer.visualize_views("test", all_imgs_list)
            trainer.update_validation_accuracy(1)
        if setup["test_only_retr"]:
            trainer.train_loader = DataLoader(dset_train, batch_size=int(setup["batch_size"]/2),
                                                             shuffle=False, num_workers=6, collate_fn=collate_fn, drop_last=True)
            trainer.update_retrieval()
        if setup["occlusion_robustness_mode"]:
            trainer.update_occlusion_robustness()
        if setup["rotation_robustness_mode"]:
            trainer.update_rotation_robustness()
    if setup["log_metrics"]:
        trainer.writer.close()


all_imgs_list = [55, 66, 77]
view_gcn_exp(setup, models_bag)
