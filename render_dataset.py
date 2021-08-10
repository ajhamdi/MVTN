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
import imageio
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
from models.pointnet import *


# from logger import Logger
from torch.utils.tensorboard import SummaryWriter
from custom_dataset import RenderingDataset, collate_fn  # , ModelNet40



def initialize_setup(setup):
    SHAPE_FEATURES_SIZE = {"logits": 40, "post_max": 1024,
                           "transform_matrix": 64*64, "pre_linear": 512, "post_max_trans": 1024 + 64*64, "logits_trans": 40+64*64, "pre_linear_trans": 512+64*64}

    setup["features_size"] = SHAPE_FEATURES_SIZE[setup["features_type"]]
    if setup["exp_id"] == "random":
        setup["exp_id"] = random_id()
    setup["results_dir"] = os.path.join(setup["GCN_dir"], setup["results_dir"])
    setup["results_dir"] = os.path.join(setup["results_dir"], setup["exp_id"])
    setup["cameras_dir"] = os.path.join(
        setup["results_dir"], setup["cameras_dir"])
    setup["renderings_dir"] = os.path.join(
        setup["results_dir"], setup["renderings_dir"])
    setup["verts_dir"] = os.path.join(setup["results_dir"], "verts")
    setup["checkpoint_dir1"] = os.path.join(
        setup["results_dir"], "checkpoint_stage1")
    setup["checkpoint_dir2"] = os.path.join(
        setup["results_dir"], "checkpoint_stage2")
    setup["train_path"] = os.path.join(setup["GCN_dir"], setup["train_path"])
    setup["val_path"] = os.path.join(setup["GCN_dir"], setup["val_path"])
    setup["cnn_name"] = "resnet{}".format(setup["depth"])
    setup["logs_dir"] = os.path.join(setup["results_dir"], setup["logs_dir"])

    # check_folder(setup["results_dir"])
    # check_folder(setup["cameras_dir"])
    # check_folder(setup["renderings_dir"])
    # check_folder(setup["logs_dir"])
    # check_folder(setup["verts_dir"])
    # check_folder(setup["checkpoint_dir1"])
    # check_folder(setup["checkpoint_dir2"])
    # shutil.copyfile(os.path.join(setup["results_dir"], "..", "model-00029.pth"),
    #                 os.path.join(setup["checkpoint_dir1"], "model-00029.pth"))
    setup["best_acc"] = 0.0
    setup["best_loss"] = 0.0
    setup["start_epoch"] = 0
    setup["results_file"] = os.path.join(
        setup["results_dir"], setup["exp_id"]+"_accuracy.csv")
    setup["views_file"] = os.path.join(
        setup["results_dir"], setup["exp_id"]+"_views.csv")
    setup["weights_file1"] = os.path.join(
        setup["checkpoint_dir1"], setup["exp_id"]+"_checkpoint.pt")
    setup["weights_file2"] = os.path.join(
        setup["checkpoint_dir2"], setup["exp_id"]+"_checkpoint.pt")
    setup["is_learning_views"] = setup["selection_type"] in [
        "learned_offset", "learned_direct", "learned_spherical", "learned_random"]
    setup["is_learning_points"] = setup["is_learning_views"] and (
        setup["return_points_saved"] or setup["return_points_sampled"])
    for k, v in setup.items():
        if isinstance(v, bool):
            setup[k] = int(v)


parser = argparse.ArgumentParser(description='ViewGCN-MVT')
parser.add_argument('--depth', choices=[18, 34, 50, 101, 152],
                    type=int,  default=18, help='resnet depth (default: resnet18)')
parser.add_argument('--gpu', type=int,
                    default=0, help='GPU number ')
# parser.add_argument('--mvnetwork', '-m',  default=RESNET, choices=NETWORKS,
#                     help='pretrained mvnetwork: ' + ' | '.join(NETWORKS) + ' (default: {})'.format(RESNET))
parser.add_argument('--epochs', default=100, type=int,
                    help='number of total epochs to run (default: 100)')
parser.add_argument('-b', '--batch_size', default=16, type=int,
                    help='mini-batch size (default: 4)')

parser.add_argument('--image_data', required=True,  help='path to 2D dataset')
parser.add_argument('--mesh_data', required=True,  help='path to 3D dataset')
parser.add_argument('--exp_id', type=str, default='random', help='pick ')
parser.add_argument('--nb_views', default=12, type=int,
                    help='number of views in MV CNN')
parser.add_argument('--nb_points', default=2048, type=int,
                    help='number of points in the 3d dataeset sampled from the meshes ')
parser.add_argument('--image_size', default=224, type=int,
                    help='the size of the images rendered by the differntibe renderer')
parser.add_argument('--canonical_elevation', default=30.0, type=float,
                    help='if selection_type== canoncal , the elevation of the view points is givene by this angle')
parser.add_argument('--canonical_distance', default=2.2, type=float,
                    help='the distnace of the view points from the center if the object  ')
parser.add_argument('--input_view_noise', default=0.0, type=float,
                    help='the variance of the gaussian noise (before normalization with parametre range) added to the azim,elev,dist inputs to the MVT ... this option is valid only if `learned_offset` or `learned_direct` options are sleected   ')
parser.add_argument('--selection_type', '-s',  default="canonical", choices=["canonical", "random", "learned_offset", "learned_direct", "spherical", "learned_spherical", "learned_random"],
                    help='the selection type of views ')
parser.add_argument('--plot_freq', default=2, type=int,
                    help='the frequqency of plotting the renderings and camera positions')
parser.add_argument('--GCN_dir', '-gcnd',  default="viewGCN",
                    help='the view-GCN folder')
parser.add_argument('--renderings_dir', '-rd',  default="renderings",
                    help='the destinatiojn for the renderings ')
parser.add_argument('--results_dir', '-rsd',  default="results",
                    help='the destinatiojn for the results ')
parser.add_argument('--logs_dir', '-lsd',  default="logs",
                    help='the destinatiojn for the tensorboard logs ')
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
parser.add_argument('--shape_extractor', '-pnet',  default="MVCNN", choices=["PointNet", "DGCNN", "PointNetPP", "MVCNN"],
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
parser.add_argument('--vs_clip_grads_value', default=3, type=float,
                    help='the clip value for L2 of gradeients of the MVT ')
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
parser.add_argument('--pretrained', dest='pretrained',
                    action='store_true', help='use pre-trained mvnetwork')
parser.add_argument('--save_all', dest='save_all',
                    action='store_true', help='save save the checkpoint and results at every epoch.... default saves only best test accuracy epoch')
parser.add_argument('--test_only', dest='test_only',
                    action='store_true', help='do only testing once ... no training ')
parser.add_argument('--return_points_sampled', dest='return_points_sampled',
                    action='store_true', help='reuturn 3d point clouds from the data loader sampled from hte mesh ')
parser.add_argument('--return_points_saved', dest='return_points_saved',
                    action='store_true', help='reuturn 3d point clouds from the data loader saved under `filePOINTS.pkl` ')
parser.add_argument('--return_extracted_features', dest='return_extracted_features',
                    action='store_true', help='return pre extracted features `*_PFeatures.pt` for each 3d model from the dataloader ')
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


parser.add_argument("---name", type=str,
                    help="Name of the experiment", default="view-gcn")
# it will be *12 images in each batch for mvcnn
parser.add_argument("--second_stage_bs", type=int,
                    help="Batch size for the second stage", default=20)
parser.add_argument("--second_stage_epochs", type=int,
                    help="number of epochs for the second stage", default=25)
parser.add_argument("--first_stage_bs", type=int,
                    help="Batch size for the first stage", default=400)
parser.add_argument("--first_stage_epochs", type=int,
                    help="number of epochs for the first stage", default=30)
parser.add_argument("--num_models", type=int,
                    help="number of models per class", default=0)
parser.add_argument("--learning_rate", type=float,
                    help="learning rate second stage", default=1e-4)
parser.add_argument("--learning_rate_first", type=float,
                    help="learning rate for first stage", default=1e-4)

parser.add_argument("--weight_decay", type=float,
                    help="weight decay", default=0.001)
# parser.add_argument("--cnn_name", "--cnn_name", type=str, help="cnn model name", default="resnet18")
parser.add_argument("--train_path", type=str,
                    default="data/modelnet40v2png_ori4/*/train")
parser.add_argument("--val_path", type=str,
                    default="data/modelnet40v2png_ori4/*/test")
parser.add_argument("--new_train_path", type=str,
                    default="viewGCN/data/modelnet40v3/*/train")
parser.add_argument("--new_val_path", type=str,
                    default="viewGCN/data/modelnet40v3/*/test")
parser.add_argument("--phase", default="all", choices=["all", "first", "second", "third"],
                    help='what stage of train/test of the VIEW-GCN ( MVCNN, or the gcn or the MVT)')


parser.set_defaults(train=False)

def create_folder(log_dir):
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    # else:
    #     print('WARNING: summary folder already exists!! It will be overwritten!!')
    #     shutil.rmtree(log_dir)
    #     os.mkdir(log_dir)

if __name__ == '__main__':
    seed_torch()
    args = parser.parse_args()
    setup = vars(args)
    initialize_setup(setup)
    n_models_train = setup["num_models"]*setup["nb_views"]
    train_dataset = SingleImgDataset(setup["train_path"], scale_aug=False, rot_aug=False, num_models=n_models_train, num_views=setup["nb_views"])
    val_dataset = SingleImgDataset(setup["val_path"], scale_aug=False, rot_aug=False, test_mode=True)
    print(len(train_dataset))
    for ii in range(len(train_dataset)):
        (class_id, im, path) = train_dataset[ii]
        path.split('/')[]
        print(path, im.shape)
viewGCN/data/modelnet40v2png_ori4/bed/train/bed_0374_013.png
