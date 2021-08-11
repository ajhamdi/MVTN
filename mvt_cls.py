import torch
import torch.nn as nn
import sys
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.autograd import Variable

import warnings
import sklearn	
import sklearn.metrics	
from sklearn.metrics import precision_recall_fscore_support
import scipy	
import scipy.spatial		
# warnings.filterwarnings("error")	
# torch.multiprocessing.set_start_method('spawn')	
from tqdm import tqdm	
import pickle as pkl



import torchvision.transforms as transforms
import torchvision

import argparse
import numpy as np
import time
import os

# from models.resnet import *
# from models.mvcnn import *
from models.pointnet import *
from util import *
from ops import *


# from logger import Logger
from torch.utils.tensorboard import SummaryWriter
from custom_dataset import MultiViewDataSet, ThreeMultiViewDataSet, collate_fn, ShapeNetCore, ScanObjectNN  # , ModelNet40
from rotationNet.mvt_rotnet import RotationNet, AverageMeter, my_accuracy
from viewGCN.tools.Trainer_mvt import ModelNetTrainer_mvt
from viewGCN.tools.ImgDataset import MultiviewImgDataset, SingleImgDataset
from viewGCN.model.view_gcn import view_GCN, SVCNN

PLOT_SAMPLE_NBS = [242,7, 549,112,34]






parser = argparse.ArgumentParser(description='MVCNN-PyTorch')
parser.add_argument('--depth', choices=[18, 34, 50, 101, 152], type=int,  default=18, help='resnet depth (default: resnet18)')
parser.add_argument('--gpu', type=int,
                     default=0, help='GPU number ')
parser.add_argument('--mvnetwork', '-m',  default="resnet", choices=["mvcnn","rotnet","viewgcn"],
                    help='the type of multi-view network used:')
parser.add_argument('--epochs', default=100, type=int,  help='number of total epochs to run (default: 100)')
parser.add_argument('-b', '--batch_size', default=20, type=int,
                     help='mini-batch size (default: 4)')

parser.add_argument('--image_data',required=False,  help='path to 2D dataset')
parser.add_argument('--mesh_data', required=True,  help='path to 3D dataset')
parser.add_argument('--exp_set', type=str, default='00', help='pick ')
parser.add_argument('--exp_id', type=str, default='random', help='pick ')
parser.add_argument('--nb_views', default=4, type=int, 
                    help='number of views in MV CNN')
parser.add_argument('--image_size', default=224, type=int, 
                    help='the size of the images rendered by the differntibe renderer')
parser.add_argument('--canonical_elevation', default=30.0, type=float,
                     help='if selection_type== canoncal , the elevation of the view points is givene by this angle')
parser.add_argument('--canonical_distance', default=2.2, type=float,
                     help='the distnace of the view points from the center if the object  ')
parser.add_argument('--input_view_noise', default=0.0, type=float,
                    help='the variance of the gaussian noise (before normalization with parametre range) added to the azim,elev,dist inputs to the MVTN ... this option is valid only if `learned_offset` or `learned_direct` options are sleected   ')
parser.add_argument('--selection_type', '-s',  default="circular", choices=["circular", "random", "learned_offset", "learned_direct", "spherical", "learned_spherical", "learned_random", "learned_transfer", "custom"],
                    help='the selection type of views ')
parser.add_argument('--plot_freq', default=3, type=int, 
                    help='the frequqency of plotting the renderings and camera positions')
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
                    action='store_true', help='use simplified meshes in learning .. not the full meshes ... it ends in `_SMPLER.obj` ')
parser.add_argument('--cleaned_mesh', dest='cleaned_mesh',
                    action='store_true', help='use cleaned meshes using reversion of light direction for faulted meshes')

parser.add_argument('--view_reg', default=0.0, type=float,
                    help='use regulizer to the learned view selector so they can be apart ...ONLY when `selection_type` == learned_direct   (default: 0.0)')
parser.add_argument('--shape_extractor', '-pnet',  default="MVCNN", choices=["PointNet", "DGCNN", "PointNetPP","MVCNN"],
                    help='pretrained point cloud mvnetwork to get coarse featrures ')
parser.add_argument('--features_type', '-ftpe',  default="post_max", choices=["logits", "post_max", "transform_matrix",
                                                                              "pre_linear", "logits_trans", "post_max_trans", "pre_linear_trans"],
                    help='the type of the features extracted from the feature extractor ( early , middle , late) ')
parser.add_argument('--transform_distance', dest='transform_distance',
                    action='store_true', help='transform the distance to the object as well  ')
parser.add_argument('--vs_weight_decay', default=0.01, type=float,
                    help='weight decay for MVTN ... default 0.01')
parser.add_argument('--clip_grads', dest='clip_grads',
                    action='store_true', help='clip the gradients of the MVTN with L2= `clip_grads_value` ')
parser.add_argument('--vs_clip_grads_value', default=30.0, type=float,help='the clip value for L2 of gradeients of the MVTN ')
parser.add_argument('--fe_clip_grads_value', default=30.0, type=float,
                    help='the clip value for L2 of gradeients of the Shape feature extractor .. eg.e PointENt ')


parser.add_argument('-lr', '--learning_rate', default=0.001, type=float,
                     help='initial learning rate (default: 0.0001)')
parser.add_argument('--weight_decay', default=0.01, type=float,
                    help='weight decay for MVTN ... default 0.01')
parser.add_argument('--momentum', default=0.9, type=float, 
                    help='momentum (default: 0.9)')
parser.add_argument('--lr-decay-freq', default=30, type=float,
                     help='learning rate decay (default: 30)')
parser.add_argument('--lr-decay', default=0.1, type=float,
                     help='learning rate decay (default: 0.1)')
parser.add_argument('--print_freq', '-p', default=50, type=int,
                     help='print frequency (default: 10)')
parser.add_argument('-r', '--resume', dest='resume',
                    action='store_true', help='continue training from the `setup[weights_file] checkpoint ')
parser.add_argument('--pretrained', dest='pretrained', action='store_true', help='use pre-trained mvnetwork')
parser.add_argument('--save_all', dest='save_all',
                    action='store_true', help='save save the checkpoint and results at every epoch.... default saves only best test accuracy epoch')
parser.add_argument('--test_only', dest='test_only',
                    action='store_true', help='do only testing once ... no training ')
parser.add_argument('--train', dest='train',
                    action='store_true', help='do only training  once ... no testing ')
parser.add_argument('--test_only_retr', dest='test_only_retr',
                    action='store_true', help='do only testing once including metrics for retrieval ... no training ')
parser.add_argument('--extract_features_mode', dest='extract_features_mode',
                    action='store_true', help='extract shape features from the 3d models to be used in training  the view selector')
parser.add_argument('--visualize_verts_mode', dest='visualize_verts_mode',
                    action='store_true', help='do visualization ... no evaluations ')
parser.add_argument('--LFDA_dimension', dest='LFDA_dimension',
                    default=64, type=int, help='dimension for LFDA projection (0 = no projection')
parser.add_argument('--LFDA_layer', dest='LFDA_layer',
                    default=0, type=int, help='layer for LFDA projection')
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
parser.add_argument('--custom_views_mode', dest='custom_views_mode',
                    action='store_true', help=' test MVCNN with `custom` views ')
# parser.add_argument('--noisy_dataset', '-ndset',  default="no", choices=["no", "low", "medium", "high"],
#                     help='the amount of rotation noise on the meshes from ModelNet40 ')
parser.add_argument('--log_metrics', dest='log_metrics',
                    action='store_true', help='logs loss and acuracy and other metrics in `logs_dir` for tensorboard ')
# parser.add_argument('--random_lighting', dest='random_lighting',
#                     action='store_true', help='apply random light direction on the rendered images .. otherwise default (0, 1.0, 0) ')
parser.add_argument('--light_direction', '-ldrct',  default="random", choices=["fixed", "random", "relative"],
                    help='apply random light direction on the rendered images .. otherwise default (0, 1.0, 0)')
parser.add_argument('--screatch_feature_extractor', dest='screatch_feature_extractor',
                    action='store_true', help='start training the feature extractor from scracth ...applies ONLY if `is_learning_points` == True ')
parser.add_argument('--late_fusion_mode', dest='late_fusion_mode',
                    action='store_true', help='start training late fusiuon of point network and MV networks ')
parser.add_argument('--cull_backfaces', dest='cull_backfaces',
                    action='store_true', help='cull back_faces ( remove them from the image) ')

## point cloud rnedienring 
parser.add_argument('--pc_rendering', dest='pc_rendering',
                    action='store_true', help='use point cloud renderer instead of mesh renderer  ')
parser.add_argument('--points_radius', default=0.006, type=float,
                    help='the size of the rendered points if `pc_rendering` is True  ')
parser.add_argument('--points_per_pixel',  default=10, type=int,
                     help='max number of points in every rendered pixel if `pc_rendering` is True ')
parser.add_argument('--dset_variant', '-dsetp',  default="obj_only", choices=["obj_only", "with_bg", "hardest"])
parser.add_argument('--nb_points', default=2048, type=int,help='number of points in the 3d dataeset sampled from the meshes ')
parser.add_argument('--object_color', '-clr',  default="white", choices=["white", "random","black","red","green","blue", "learned","custom"],
                    help='the selection type of views ')
parser.add_argument('--background_color', '-bgc',  default="white", choices=["white", "random","black","red","green","blue", "learned","custom"],
                    help='the color of the background of the rendered images')
parser.add_argument('--dset_norm', '-dstn',  default="2", choices=["inf", "2","1","fro","no"],
                    help='the L P normlization tyoe of the 3D dataset')
parser.add_argument('--initial_angle', default=-90.0, type=float,
                    help='the inital tilt angle of the 3D object as loaded from the ModelNet40 dataset  ')                  
parser.add_argument('--occlusion_robustness_mode', dest='occlusion_robustness_mode',
                    action='store_true', help='use the point cloud rendering to test robustness to occlusion of chopping point cloud ')
parser.add_argument('--rotation_robustness_mode', dest='rotation_robustness_mode',
                    action='store_true', help='use the point cloud rendering to test robustness to rotation at test time')
parser.add_argument('--measure_speed_mode', dest='measure_speed_mode',
                    action='store_true', help='measure speed and memory of all differenct components of MVTN ')
parser.add_argument('--max_degs', default=180.0, type=float,
                    help='the maximum allowed Z rotation degrees on the meshes ')
parser.add_argument('--repeat_exp', '-rpx', default=3, type=int,
                    help='the number of repeated exps for each setup ( due to randomness)')
parser.add_argument('--augment_training', dest='augment_training',
                    action='store_true', help='augment the training of the CNN by scaling , rotation , translation , etc ')
parser.add_argument('--crop_ratio', default=0.3, type=float,
                    help='the crop ratio of the images when `augment_training` == True  ')  


# ViewGCN adaptation in MVTN
parser.add_argument('--GCN_dir', '-gcnd',  default="viewGCN",help='the view-GCN folder')
parser.add_argument("--phase", default="all", choices=["all", "first", "second", "third"],
                    help='what stage of train/test of the VIEW-GCN ( CNN, or the gcn or the MVTN)')
parser.add_argument('--ignore_normalize', dest='ignore_normalize',
                    action='store_true', help='ignore any normalization performed on the image (mean,std) before passing to the network')
parser.add_argument('--resume_mvtn', dest='resume_mvtn',
                    action='store_true', help='use a pretrained MVTN and freez during this training  ')
parser.add_argument("---name", type=str,
                    help="Name of the experiment", default="view-gcn")
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
parser.add_argument('-rr', '--resume_first', dest='resume_first',
                    action='store_true', help='continue training from the `setup[weights_file] checkpoint ')
parser.add_argument('-rrr', '--resume_second', dest='resume_second',
                    action='store_true', help='continue training from the `setup[weights_file] checkpoint ')
args = parser.parse_args()
setup = vars(args)
if setup["mvnetwork"] in ["rotnet","mvcnn"]:
    initialize_setup(setup)
else:
    initialize_setup_gcn(setup)

print('Loading data')

transform = transforms.Compose([
    transforms.CenterCrop(500),
    transforms.Resize(224),
    transforms.ToTensor(),
])
# a function to preprocess pytorch3d Mesh onject

# device = torch.device("cuda:{}".format(str(setup["gpu"])) if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(int(setup["gpu"]))
if "modelnet" in setup["mesh_data"].lower():
    dset_train = ThreeMultiViewDataSet(
        'train', setup, transform=transform, is_rotated=setup["rotated_train"])
    dset_val = ThreeMultiViewDataSet(
    'test', setup, transform=transform, is_rotated=setup["rotated_test"])
    classes = dset_train.classes

elif "shapenetcore" in setup["mesh_data"].lower():
    dset_train = ShapeNetCore(setup["mesh_data"],("train",), setup["nb_points"], load_textures=False, dset_norm=setup["dset_norm"],simplified_mesh=setup["simplified_mesh"])
    dset_val = ShapeNetCore(setup["mesh_data"],("test",), setup["nb_points"], load_textures=False, dset_norm=setup["dset_norm"],simplified_mesh=setup["simplified_mesh"])
    # dset_train, dset_val = torch.utils.data.random_split(shapenet, [int(.8*len(shapenet)), int(np.ceil(0.2*len(shapenet)))])  #, generator=torch.Generator().manual_seed(42))   ## to reprodebel results 
    classes = dset_val.classes
elif "scanobjectnn" in setup["mesh_data"].lower():
    dset_train = ScanObjectNN(setup["mesh_data"], 'train',  setup["nb_points"],
                              variant=setup["dset_variant"], dset_norm=setup["dset_norm"])
    dset_val = ScanObjectNN(setup["mesh_data"], 'test',  setup["nb_points"], variant=setup["dset_variant"], dset_norm=setup["dset_norm"])
    classes = dset_train.classes

train_loader = DataLoader(dset_train, batch_size=setup["batch_size"],
                          shuffle=True, num_workers=6, collate_fn=collate_fn, drop_last=True)

val_loader = DataLoader(dset_val, batch_size=int(setup["batch_size"]/2),
                        shuffle=False, num_workers=6, collate_fn=collate_fn)

print("classes nb:", len(classes), "number of train models: ", len(
    dset_train), "number of test models: ", len(dset_val), classes)

if setup["mvnetwork"] == "mvcnn":
    depth2featdim = {18: 512 ,34: 512,50: 2048,101: 2048,152: 2048 }
    assert setup["depth"] in list(depth2featdim.keys()) , "the requested resnt depth not available"
    mvnetwork = torchvision.models.__dict__["resnet{}".format(setup["depth"])](setup["pretrained"])
    mvnetwork.fc = nn.Sequential()
    mvnetwork = MVAgregate(mvnetwork, agr_type="max", feat_dim=depth2featdim[setup["depth"]], num_classes=len(classes))
    print('Using ' + setup["mvnetwork"] + str(setup["depth"]))
if setup["mvnetwork"] == "rotnet":
    mvnetwork = torchvision.models.__dict__["resnet{}".format(setup["depth"])](pretrained=setup["pretrained"])
    mvnetwork = RotationNet(mvnetwork, "resnet{}".format(setup["depth"]), (len(classes)+1) * setup["nb_views"])
if setup["mvnetwork"] == "viewgcn":
    mvnetwork = SVCNN(setup["exp_id"], nclasses=len(classes),pretraining=setup["pretrained"], cnn_name=setup["cnn_name"])

mvnetwork.cuda()
cudnn.benchmark = True

print('Running on ' + str(torch.cuda.current_device()))


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
optimizer = torch.optim.AdamW(
    mvnetwork.parameters(), lr=lr, weight_decay=setup["weight_decay"])
if setup["is_learning_views"]:
    vs_optimizer = torch.optim.AdamW(view_selector.parameters(), lr=setup["vs_learning_rate"], weight_decay=setup["vs_weight_decay"])
else : 
    vs_optimizer = None
if setup["is_learning_points"]:
    fe_optimizer = torch.optim.AdamW(feature_extractor.parameters(), lr=setup["pn_learning_rate"])
else:
    fe_optimizer = None

models_bag = {"mvnetwork": mvnetwork,
              "optimizer": optimizer, "view_selector": view_selector, "vs_optimizer": vs_optimizer, "fe_optimizer": fe_optimizer,
              "feature_extractor": feature_extractor}



def train(data_loader, models_bag, setup):
    train_size = len(data_loader)
    total = 0.0
    correct = 0.0

    total_loss = 0.0
    n = 0

    for i, (targets, meshes, extra_info,correction_factor) in enumerate(data_loader):
        models_bag["optimizer"].zero_grad()
        if setup["is_learning_views"]:
            models_bag["vs_optimizer"].zero_grad()
        if setup["is_learning_points"]:
            models_bag["fe_optimizer"].zero_grad()

        # inputs = np.stack(inputs, axis=1)
        # inputs = torch.from_numpy(inputs)
        rendered_images, _, azim, elev, dist = auto_render_meshes(
            targets, meshes, extra_info,correction_factor, models_bag, setup, device=None)

        targets = targets.cuda()
        targets = Variable(targets)
        outputs = models_bag["mvnetwork"](rendered_images)[0]
        loss = criterion(outputs, targets)
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)

        # compute gradient and do SGD step
        # print("\n\n\nvals:        {:.2f} , {:.2f} , {:.2f}".format(azim[0, 0].item(), elev[0, 0].item(), dist[0, 0].item()))
        # print("VS ang grads:        ", azim.grad[0, 0].item(), elev.grad[0, 0].item(), dist.grad[0, 0].item())

        loss.backward()


        # print("net grads:      ", list(mvnetwork.parameters())[0].data[0, 0, 0, 0].item())
        # print("\n MVNETWORK grads:      ", np.sum(np.array([np.sum(x.grad.cpu().numpy() ** 2) for x in mvnetwork.parameters()])))

        models_bag["optimizer"].step()
        if setup["is_learning_views"]:
            models_bag["vs_optimizer"].step()
            if setup["clip_grads"]:
                clip_grads_(models_bag["view_selector"].parameters(), setup["vs_clip_grads_value"])
            if setup["log_metrics"]:
                step = get_current_step(models_bag["vs_optimizer"])
                writer.add_scalar('Zoom/loss', loss.item(), step)
                writer.add_scalar('Zoom/MVT_vals', list(models_bag["view_selector"].parameters())[0].data[0, 0].item(), step)
                writer.add_scalar('Zoom/MVT_grads', np.sum(np.array([np.sum(x.grad.cpu().numpy() ** 2) for x in models_bag["view_selector"].parameters()])), step)
                writer.add_scalar('Zoom/MVCNN_vals', list(models_bag["mvnetwork"].parameters())[0].data[0].item(), step)
                writer.add_scalar('Zoom/MVCNN_grads', np.sum(np.array([np.sum(x.grad.cpu().numpy() ** 2) for x in models_bag["mvnetwork"].parameters()])), step)
        if setup["is_learning_points"]:
            models_bag["fe_optimizer"].step()
            if setup["clip_grads"]:
                clip_grads_(models_bag["feature_extractor"].parameters(),setup["fe_clip_grads_value"])
            if setup["log_metrics"]:
                step = get_current_step(models_bag["fe_optimizer"])
                writer.add_scalar('Zoom/PNet_vals', list(filter(lambda p: p.grad is not None, list(models_bag["feature_extractor"].parameters())))[0].data[0, 0].item(), step)
                writer.add_scalar('Zoom/PNet_grads', np.sum(np.array([np.sum(x.grad.cpu(
                ).numpy() ** 2) for x in list(filter(lambda p: p.grad is not None, list(models_bag["feature_extractor"].parameters())))])), step)



        if (i + 1) % setup["print_freq"] == 0:
            print("\tIter [%d/%d] Loss: %.4f" % (i + 1, train_size, loss.item()))
        correct += (predicted.cpu() == targets.cpu()).sum()
        total_loss += loss
        n += 1
    avg_loss = total_loss / n
    avg_train_acc = 100 * correct / total

    return avg_train_acc,avg_loss


def train_rotationNet(data_loader, models_bag, setup):
    train_size = len(data_loader)

    total_loss = 0.0
    n = 0
    top1 = AverageMeter()


    for i, (targets, meshes, extra_info, correction_factor) in enumerate(data_loader):
        models_bag["optimizer"].zero_grad()
        if setup["is_learning_views"]:
            models_bag["vs_optimizer"].zero_grad()
        if setup["is_learning_points"]:
            models_bag["fe_optimizer"].zero_grad()

        # inputs = np.stack(inputs, axis=1)
        # inputs = torch.from_numpy(inputs)
        c_bs = targets.shape[0]
        rendered_images, _, azim, elev, dist = auto_render_meshes(targets, meshes, extra_info, correction_factor, models_bag, setup, device=None)

        targets = targets.repeat_interleave((setup["nb_views"])).cuda()

        input_var = mvctosvc(rendered_images).cuda()
        targets_ = torch.LongTensor(targets.size(0) * setup["nb_views"])

        # compute output
        output = models_bag["mvnetwork"](input_var)
        num_classes = int(output.size(1) / setup["nb_views"]) - 1
        output = output.view(-1, num_classes + 1)

        ###########################################
        # compute scores and decide targets labels #
        ###########################################
        output_ = torch.nn.functional.log_softmax(output,dim=-1)
        # divide object scores by the scores for "incorrect view label" (see Eq.(5))
        output_ = output_[
            :, :-1] - torch.t(output_[:, -1].repeat(1, output_.size(1)-1).view(output_.size(1)-1, -1))
        # reshape output matrix
        output_ = output_.view(-1, setup["nb_views"] * setup["nb_views"], num_classes)
        output_ = output_.data.cpu().numpy()
        output_ = output_.transpose(1, 2, 0)
        # initialize targets labels with "incorrect view label"
        for j in range(targets_.size(0)):
            targets_[j] = num_classes
        # compute scores for all the candidate poses (see Eq.(5))
        scores = np.zeros((vcand.shape[0], num_classes, c_bs))
        for j in range(vcand.shape[0]):
            for k in range(vcand.shape[1]):
                scores[j] = scores[j] + output_[vcand[j][k] * setup["nb_views"] + k]
        # for each sample #n, determine the best pose that maximizes the score for the targets class (see Eq.(2))
        for n in range(c_bs):
            j_max = np.argmax(scores[:, targets[n * setup["nb_views"]], n])
            # assign targets labels
            for k in range(vcand.shape[1]):
                targets_[n * setup["nb_views"] * setup["nb_views"] + vcand[j_max]
                        [k] * setup["nb_views"] + k] = targets[n * setup["nb_views"]]
        ###########################################

        targets_ = targets_.cuda()
        targets_var = torch.autograd.Variable(targets_)

        # compute loss
        loss = criterion(output, targets_var)





        # outputs = models_bag["mvnetwork"](rendered_images)[0]
        # loss = criterion(outputs, targets)
        # _, predicted = torch.max(output.data, 1)
        # print(output.shape)

        # total += targets.size(0)
        loss.backward()

        models_bag["optimizer"].step()
        if setup["is_learning_views"]:
            models_bag["vs_optimizer"].step()
            if setup["clip_grads"]:
                clip_grads_(models_bag["view_selector"].parameters(), setup["vs_clip_grads_value"])
            if setup["log_metrics"]:
                step = get_current_step(models_bag["vs_optimizer"])
                writer.add_scalar('Zoom/loss', loss.item(), step)
                writer.add_scalar(
                    'Zoom/MVT_vals', list(models_bag["view_selector"].parameters())[0].data[0, 0].item(), step)
                writer.add_scalar('Zoom/MVT_grads', np.sum(np.array([np.sum(x.grad.cpu(
                ).numpy() ** 2) for x in models_bag["view_selector"].parameters()])), step)
                writer.add_scalar(
                    'Zoom/MVCNN_vals', list(models_bag["mvnetwork"].parameters())[0].data[0, 0, 0, 0].item(), step)
                writer.add_scalar('Zoom/MVCNN_grads', np.sum(np.array([np.sum(
                    x.grad.cpu().numpy() ** 2) for x in models_bag["mvnetwork"].parameters()])), step)
        if setup["is_learning_points"]:
            models_bag["fe_optimizer"].step()
            if setup["clip_grads"]:
                clip_grads_(models_bag["feature_extractor"].parameters(), setup["fe_clip_grads_value"])
            if setup["log_metrics"]:
                step = get_current_step(models_bag["fe_optimizer"])
                writer.add_scalar('Zoom/PNet_vals', list(filter(lambda p: p.grad is not None, list(
                    models_bag["feature_extractor"].parameters())))[0].data[0, 0].item(), step)
                writer.add_scalar('Zoom/PNet_grads', np.sum(np.array([np.sum(x.grad.cpu(
                ).numpy() ** 2) for x in list(filter(lambda p: p.grad is not None, list(models_bag["feature_extractor"].parameters())))])), step)
        output = output[:, :-1] - torch.t(output[:, -1].repeat(1, output.size(1)-1).view(output.size(1)-1, -1))
        output = output.view(-1, setup["nb_views"] * setup["nb_views"], num_classes)
        prec1,_ = my_accuracy(output.data, targets,vcand,setup["nb_views"], topk=(1,5))
        top1.update(prec1.item(), c_bs)


        if (i + 1) % setup["print_freq"] == 0:
            print("\tIter [%d/%d] Loss: %.4f" %
                  (i + 1, train_size, loss.item()))
        # correct += (predicted.cpu() == targets.cpu()).sum()
        total_loss += loss
        n += 1
    avg_loss = total_loss / n
    # avg_train_acc = 100 * correct / total

    return top1.avg, avg_loss


def evaluate_rotationNet(data_loader, models_bag, setup):
    train_size = len(data_loader)

    total_loss = 0.0
    n = 0
    top1 = AverageMeter()
    t = tqdm(enumerate(data_loader),total=len(data_loader))
    for i, (targets, meshes, extra_info, correction_factor) in t:
        with torch.no_grad():

            c_bs = targets.shape[0]
            rendered_images, _, azim, elev, dist = auto_render_meshes(
                targets, meshes, extra_info, correction_factor, models_bag, setup, device=None)

            targets = targets.repeat_interleave((setup["nb_views"])).cuda()

            input_var = torch.autograd.Variable(mvctosvc(rendered_images)).cuda()
            target_var = torch.autograd.Variable(targets)

            # compute output
            output = models_bag["mvnetwork"](input_var)
            loss = criterion(output, target_var)

            # log_softmax and reshape output
            num_classes = int(output.size(1) / setup["nb_views"]) - 1
            output = output.view(-1, num_classes + 1)
            output = torch.nn.functional.log_softmax(output,dim=-1)
            output = output[:, :-1] - torch.t(output[:, -1].repeat(
                1, output.size(1)-1).view(output.size(1)-1, -1))
            output = output.view(-1, setup["nb_views"] * setup["nb_views"], num_classes)
            output = output.view(-1, setup["nb_views"]
                                * setup["nb_views"], num_classes)
            prec1, _ = my_accuracy(output.data, targets, vcand,
                                setup["nb_views"], topk=(1, 5))
            top1.update(prec1.item(), c_bs)

            total_loss += loss
            n += 1
    avg_loss = total_loss / n

    return top1.avg, avg_loss

# Validation and Testing
def evluate(data_loader, models_bag,  setup, is_test=False, retrieval=False):
    if is_test:
        load_checkpoint(setup, models_bag, setup["weights_file"])
    
    # Eval
    total = 0.0
    correct = 0.0

    total_loss = 0.0
    n = 0
    if retrieval:
        features_training = np.load(setup["feature_file"])	
        targets_training = np.load(setup["targets_file"])	
        N_retrieved = 1000 if "shapenetcore" in setup["mesh_data"].lower() else len(features_training)

        features_training = lfda.transform(features_training)	
        # print("features_training.shape [training]", features_training.shape, targets_training.shape)	
        # from pykdtree.kdtree import KDTree	

        kdtree = scipy.spatial.KDTree(features_training)	
        all_APs = []

    views_record = ListDict(["azim", "elev", "dist","label","view_nb","exp_id"])
    t = tqdm(enumerate(data_loader), total=len(data_loader))	
    for i, (targets, meshes, extra_info,correction_factor) in t:
    # for i, (targets, meshes, extra_info,correction_factor) in enumerate(data_loader):
        with torch.no_grad():
            # inputs = np.stack(inputs, axis=1)
            # inputs = torch.from_numpy(inputs)
            if setup["custom_views_mode"] :
                rendered_images, _, azim, elev, dist = auto_render_meshes_custom_views(
                targets, meshes, extra_info,correction_factor, models_bag, setup, device=None)
            else:
                rendered_images, _, azim, elev, dist = auto_render_meshes(targets, meshes, extra_info,correction_factor, models_bag, setup, device=None)
            targets = targets.cuda()
            targets = Variable(targets)
            # outputs = models_bag["mvnetwork"](rendered_images)[0]
            outputs, feat = models_bag["mvnetwork"](rendered_images) # return features as well	
            if retrieval:
                feat = feat.cpu().numpy()	
                feat = lfda.transform(feat)	
                d, idx_closest = kdtree.query(feat, k=len(features_training))	
                # loop over queries in the query	
                for i_query_batch in range(feat.shape[0]):	
                    # details on retrieval-mAP: https://towardsdatascience.com/breaking-down-mean-average-precision-map-ae462f623a52#f9ce	
                    positives = targets_training[idx_closest[i_query_batch,:]] == targets[i_query_batch].cpu().numpy()	
                    # AP: numerator is cumulative of positives, zero-ing negatives 	
                    num = np.cumsum(positives)	
                    num[~positives] = 0	
                    # AP: denominator is number of retrieved shapes	
                    den = np.array([i+1 for i in range(len(features_training))])	
                    # AP: GTP is number of positive ground truth	
                    GTP = np.sum(positives)	
                    # print(den)	
                    AP = np.sum(num/den)/GTP	
                    all_APs.append(AP)
            
            
            loss = criterion(outputs, targets)
            c_views = ListDict({"azim": azim.cpu().numpy().reshape(-1).tolist(), "elev": elev.cpu().numpy().reshape(-1).tolist(),
                       "dist": dist.cpu().numpy().reshape(-1).tolist(), "label": np.repeat(targets.cpu().numpy(), setup["nb_views"]).tolist(),
                       "view_nb": int(targets.cpu().numpy().shape[0]) * list(range(setup["nb_views"])),
                                "exp_id": int(targets.cpu().numpy().shape[0]) * int(setup["nb_views"]) * [setup["exp_id"]]  } )
            views_record.extend(c_views)
            total_loss += loss
            n += 1
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted.cpu() == targets.cpu()).sum()


    # avg_test_acc = 100 * correct / total
    # avg_loss = total_loss / n

    avg_loss = total_loss / n	
    avg_test_acc = 100 * correct / total	
    if retrieval:
        retr_map = 100 * sum(all_APs)/len(all_APs)	
        print("avg_loss", avg_loss)    	
        print("avg_test_acc", avg_test_acc)	
        print("retr_map", retr_map)	
        return avg_test_acc, retr_map, avg_loss, views_record

    return avg_test_acc, avg_loss, views_record

def compute_features(data_loader, models_bag, setup):	
    # if is_test:	
        # load_checkpoint(setup, models_bag, setup["weights_file"])	
    print("compute training metrics and store training features")	
    # Eval	
    total = 0.0	
    correct = 0.0	
    total_loss = 0.0	
    n = 0	
    feat_list=[]	
    target_list=[]	
    views_record = ListDict(["azim", "elev", "dist","label","view_nb","exp_id"])	
    t = tqdm(enumerate(data_loader), total=len(data_loader))	
    for i, (targets, meshes, extra_info,correction_factor) in t:	
        with torch.no_grad():	
            # if i > 5: break	
            # inputs = np.stack(inputs, axis=1)	
            # inputs = torch.from_numpy(inputs)	
            rendered_images, _, azim, elev, dist = auto_render_meshes(	
                targets, meshes, extra_info,correction_factor, models_bag, setup, device=None)	
            targets = targets.cuda()	
            targets = Variable(targets)	
            outputs, feat = models_bag["mvnetwork"](rendered_images)	
            	
            feat_list.append(feat.cpu().numpy())	
            target_list.append(targets.cpu().numpy())	
            	
            loss = criterion(outputs, targets)	
            c_views = ListDict({"azim": azim.cpu().numpy().reshape(-1).tolist(), "elev": elev.cpu().numpy().reshape(-1).tolist(),	
                       "dist": dist.cpu().numpy().reshape(-1).tolist(), "label": np.repeat(targets.cpu().numpy(), setup["nb_views"]).tolist(),	
                       "view_nb": int(targets.cpu().numpy().shape[0]) * list(range(setup["nb_views"])),	
                                "exp_id": int(targets.cpu().numpy().shape[0]) * int(setup["nb_views"]) * [setup["exp_id"]]  } )	
            views_record.extend(c_views)	
            total_loss += loss	
            n += 1	
            _, predicted = torch.max(outputs.data, 1)	
            total += targets.size(0)	
            correct += (predicted.cpu() == targets.cpu()).sum()	
            t.set_description(f"{i} - Acc {100 * correct / total :2.2f} - Loss {total_loss / n:2.6f}")	
    features = np.concatenate(feat_list)	
    targets = np.concatenate(target_list)	
    avg_test_acc = 100 * correct / total	
    avg_loss = total_loss / n	
    return features, targets


def evluate_late_fusion(data_loader, models_bag,  setup,):

    # Eval
    total = 0.0
    correct = 0.0

    total_loss = 0.0
    n = 0

    views_record = ListDict(
        ["azim", "elev", "dist", "label", "view_nb", "exp_id"])
    t = tqdm(enumerate(data_loader), total=len(data_loader))
    for i, (targets, meshes, extra_info,correction_factor) in t:
        c_batch = len(meshes)
        with torch.no_grad():

            rendered_images, _, azim, elev, dist = auto_render_meshes(
                targets, meshes, extra_info,correction_factor, models_bag, setup, device=None)
            targets = targets.cuda()
            targets = Variable(targets)
            # outputs = models_bag["mvnetwork"](rendered_images)[0]
            feat, _  = models_bag["mvnetwork"](
                rendered_images)
            # print(feat.shape)
            features_order = {"logits": 0,
                              "post_max": 1, "transform_matrix": 2}
            extra_info = extra_info.transpose(1, 2).cuda()
            point_features = models_bag["point_network"](extra_info)
            if setup["features_type"] == "logits_trans":
                point_features = torch.cat((point_features[0].view(
                    c_batch, -1), point_features[2].view(c_batch, -1)), 1)
            elif setup["features_type"] == "post_max_trans":
                point_features = torch.cat((point_features[1].view(c_batch, -1), point_features[2].view(c_batch, -1)), 1)
            else:
                point_features =  point_features[features_order[setup["features_type"]]].view(c_batch, -1)

            outputs = torch.max(feat, point_features)

            loss = criterion(outputs, targets)

            total_loss += loss
            n += 1
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted.cpu() == targets.cpu()).sum()

    avg_loss = total_loss / n
    avg_test_acc = 100 * correct / total

    return avg_test_acc, avg_loss, views_record


def train_late_fusion(data_loader, models_bag, setup):
    train_size = len(data_loader)
    total = 0.0
    correct = 0.0

    total_loss = 0.0
    n = 0

    for i, (targets, meshes, extra_info,correction_factor) in enumerate(tqdm(data_loader)):
        c_batch = len(meshes)

        models_bag["optimizer"].zero_grad()
        models_bag["cls_optimizer"].zero_grad()
        models_bag["fe_optimizer"].zero_grad()
        if setup["is_learning_views"]:
            models_bag["vs_optimizer"].zero_grad()
        if setup["is_learning_points"]:
            models_bag["fe_optimizer"].zero_grad()


        rendered_images, _, azim, elev, dist = auto_render_meshes(
            targets, meshes, extra_info,correction_factor, models_bag, setup, device=None)

        targets = targets.cuda()
        targets = Variable(targets)
        feat, _ = models_bag["mvnetwork"](rendered_images)
        features_order = {"logits": 0,"post_max": 1, "transform_matrix": 2}
        extra_info = extra_info.transpose(1, 2).cuda()
        point_features = models_bag["point_network"](extra_info)
        if setup["features_type"] == "logits_trans":
            point_features = torch.cat((point_features[0].view(
                c_batch, -1), point_features[2].view(c_batch, -1)), 1)
        elif setup["features_type"] == "post_max_trans":
             point_features = torch.cat((point_features[1].view(c_batch, -1), point_features[2].view(c_batch, -1)), 1)
        else:
            point_features =  point_features[features_order[setup["features_type"]]].view(c_batch, -1)

        outputs = torch.max(feat, point_features)


        loss = criterion(outputs, targets)
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)

        loss.backward()
        models_bag["optimizer"].step()
        models_bag["cls_optimizer"].step()
        models_bag["fe_optimizer"].step()

        if setup["is_learning_views"]:
            models_bag["vs_optimizer"].step()
            if setup["clip_grads"]:
                clip_grads_(models_bag["view_selector"].parameters(
                ), setup["vs_clip_grads_value"])

        if setup["is_learning_points"]:
            models_bag["fe_optimizer"].step()
            if setup["clip_grads"]:
                clip_grads_(models_bag["feature_extractor"].parameters(
                ), setup["fe_clip_grads_value"])

        if (i + 1) % setup["print_freq"] == 0:
            print("\tIter [%d/%d] Loss: %.4f" %
                  (i + 1, train_size, loss.item()))
        correct += (predicted.cpu() == targets.cpu()).sum()
        total_loss += loss
        n += 1
    avg_loss = total_loss / n
    avg_train_acc = 100 * correct / total

    return avg_train_acc, avg_loss


def evluate_rotation_robustness(data_loader, models_bag,  setup, max_degs=180.0,):
    # Eval
    total = 0.0
    correct = 0.0

    total_loss = 0.0
    n = 0
    for i, (targets, meshes, extra_info, correction_factor) in enumerate(tqdm(data_loader)):
        with torch.no_grad():

            c_batch = targets.shape[0]
            rot_axis = [0.0, 1.0, 0.0]
            angles = [np.random.rand()*20.*max_degs -
                      max_degs for _ in range(c_batch)]

            rotR = np.array([rotation_matrix(rot_axis, angle)
                             for angle in angles])
            meshes = Meshes(
                verts=[torch.mm(torch.from_numpy(rotR[ii]).to(torch.float), msh.verts_list()[
                                0].transpose(0, 1)).transpose(0, 1).cuda() for ii, msh in enumerate(meshes)],
                faces=[msh.faces_list()[0].cuda() for msh in meshes],
                textures=None)
            max_vert = meshes.verts_padded().shape[1]


            meshes.textures = Textures(verts_rgb=torch.ones(
                (c_batch, max_vert, 3)) .cuda())

            extra_info = torch.bmm(torch.from_numpy(
                rotR).to(torch.float), extra_info.transpose(1, 2)).transpose(1, 2)

            rendered_images, _, azim, elev, dist = auto_render_meshes(
                targets, meshes, extra_info, correction_factor, models_bag, setup, device=None)
            targets = targets.cuda()
            targets = Variable(targets)
            outputs = models_bag["mvnetwork"](rendered_images)[0]
            loss = criterion(outputs, targets)

            total_loss += loss
            n += 1
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted.cpu() == targets.cpu()).sum()

    avg_test_acc = 100 * correct / total
    avg_loss = total_loss / n

    return avg_test_acc, avg_loss
def visualize_retrieval_views(dtst, object_nbs,models_bag, setup, device):
    cameras_root_folder = os.path.join(
        setup["cameras_dir"], "retrieval")
    check_folder(cameras_root_folder)
    renderings_root_folder = os.path.join(
        setup["renderings_dir"], "retrieval")
    check_folder(renderings_root_folder)
    for indx, ii in enumerate(object_nbs):
        (targets, meshes, extra_info,correction_factor) = dtst[ii]
        print(targets)
        cameras_path = os.path.join(
            cameras_root_folder, "{}.jpg".format(ii))
        images_path = os.path.join(
            renderings_root_folder, "{}.jpg".format(ii))
        #extra_info = torch.from_numpy(extra_info)
        auto_render_and_save_images_and_cameras(targets, meshes, extra_info,correction_factor, images_path=images_path,
                                                cameras_path=cameras_path, models_bag=models_bag, setup=setup, device=None)

def analyze_rendered_views(dtst, object_nbs,models_bag, setup, device):
    compiled_analysis_list = []
    cameras_root_folder = os.path.join(
        setup["cameras_dir"], "retrieval")
    check_folder(cameras_root_folder)
    renderings_root_folder = os.path.join(
        setup["renderings_dir"], "retrieval")
    check_folder(renderings_root_folder)
    for indx, ii in enumerate(object_nbs):
        (targets, meshes, extra_info,correction_factor) = dtst[ii]
        # print(targets)
        cameras_path = os.path.join(
            cameras_root_folder, "{}.jpg".format(ii))
        images_path = os.path.join(
            renderings_root_folder, "{}.jpg".format(ii))
        img_avg = auto_render_and_analyze_images(targets, meshes, extra_info,correction_factor, images_path=images_path,
                                                cameras_path=cameras_path, models_bag=models_bag, setup=setup, device=None)
        compiled_analysis_list.append(img_avg)
    return compiled_analysis_list
    

# Training / Eval loop
if setup["resume"] or setup["test_only"] or setup["test_only_retr"] or setup["custom_views_mode"] or setup["occlusion_robustness_mode"] or setup["rotation_robustness_mode"]:
    if setup["mvnetwork"] in ["mvcnn","rotnet"]:
        load_checkpoint(setup, models_bag, setup["weights_file"])

if setup["mvnetwork"] == "mvcnn":
    if setup["train"] :
        if setup["log_metrics"]:
            writer = SummaryWriter(setup["logs_dir"])
        for epoch in range(setup["start_epoch"], n_epochs):
            setup["c_epoch"] = epoch
            print('\n-----------------------------------')
            print('Epoch: [%d/%d]' % (epoch+1, n_epochs))
            start = time.time()
            models_bag["mvnetwork"].train()
            models_bag["view_selector"].train()
            models_bag["feature_extractor"].train()
            avg_train_acc, avg_train_loss = train(train_loader, models_bag, setup)
            print('Time taken: %.2f sec.' % (time.time() - start))

            models_bag["mvnetwork"].eval()
            models_bag["view_selector"].eval()
            models_bag["feature_extractor"].eval()
            avg_test_acc, avg_loss, views_record = evluate(
                val_loader, models_bag, setup)

            print('\nEvaluation:')
            print('\ttrain acc: %.2f - train Loss: %.4f' %(avg_train_acc.item(), avg_train_loss.item()))
            print('\tVal Acc: %.2f - val Loss: %.4f' % (avg_test_acc.item(), avg_loss.item()))
            print('\tCurrent best val acc: %.2f' % setup["best_acc"])
            if setup["log_metrics"]:
                writer.add_scalar('Loss/train', avg_train_loss.item(), epoch)
                writer.add_scalar('Loss/val', avg_loss.item(), epoch)
                writer.add_scalar('Accuracy/train', avg_train_acc.item(), epoch)
                writer.add_scalar('Accuracy/val', avg_test_acc.item(), epoch)
            saveables = {'epoch': epoch + 1,
                        'state_dict': models_bag["mvnetwork"].state_dict(),
                        "view_selector": models_bag["view_selector"].state_dict(),
                        "feature_extractor": models_bag["feature_extractor"].state_dict(),
                        'acc': avg_test_acc,
                        'best_acc': setup["best_acc"],
                        'optimizer': models_bag["optimizer"].state_dict(),
                        'vs_optimizer': None if not setup["is_learning_views"] else models_bag["vs_optimizer"].state_dict(),
                        'fe_optimizer': None if not setup["is_learning_points"] else models_bag["fe_optimizer"].state_dict(),
                        }
            if setup["save_all"]:
                save_checkpoint(saveables, setup, views_record,setup["weights_file"])
            # Save mvnetwork
            if avg_test_acc.item() >= setup["best_acc"]:
                print('\tSaving checkpoint - Acc: %.2f' % avg_test_acc)
                saveables["best_acc"] = avg_test_acc
                setup["best_loss"] = avg_loss.item()
                setup["best_acc"] = avg_test_acc.item()
                save_checkpoint(saveables, setup, views_record,
                                setup["weights_file"])

            # Decaying Learning Rate
            if (epoch + 1) % setup["lr_decay_freq"] == 0:
                lr *= setup["lr_decay"]
                models_bag["optimizer"] = torch.optim.AdamW(
                    models_bag["mvnetwork"].parameters(), lr=lr)
                print('Learning rate:', lr)
            if (epoch + 1) % setup["plot_freq"] == 0:
                for indx,ii in enumerate( PLOT_SAMPLE_NBS):
                    (targets, meshes, extra_info,correction_factor) = dset_val[ii]
                    cameras_root_folder = os.path.join(setup["cameras_dir"],str(indx))
                    check_folder(cameras_root_folder)
                    renderings_root_folder = os.path.join(setup["renderings_dir"], str(indx))
                    check_folder(renderings_root_folder)
                    cameras_path = os.path.join(
                        cameras_root_folder, "MV_cameras_{}.jpg".format(str(epoch + 1)))
                    images_path = os.path.join(
                        renderings_root_folder, "MV_renderings_{}.jpg".format(str(epoch + 1)))
                    #extra_info = torch.from_numpy(extra_info)
                    auto_render_and_save_images_and_cameras(targets, meshes, extra_info,correction_factor, images_path=images_path,
                                                            cameras_path=cameras_path, models_bag=models_bag, setup=setup, device=None)
        if setup["log_metrics"]:
            writer.add_hparams(setup, {"hparams/best_acc": setup["best_acc"]})
    if setup["test_only"]:
        print('\nEvaluation:')
        models_bag["mvnetwork"].eval()
        models_bag["view_selector"].eval()
        models_bag["feature_extractor"].eval()

        avg_test_acc, avg_test_loss, _ = evluate(val_loader, models_bag, setup)
        print('\tVal Acc: %.2f - val Loss: %.4f' %
            (avg_test_acc.item(), avg_test_loss.item()))
        print('\tCurrent best val acc: %.2f' % setup["best_acc"])
        for indx, ii in enumerate(PLOT_SAMPLE_NBS):
            (targets, meshes, extra_info,correction_factor) = dset_val[ii]
            cameras_root_folder = os.path.join(setup["cameras_dir"],str(indx))
            check_folder(cameras_root_folder)
            renderings_root_folder = os.path.join(setup["renderings_dir"], str(indx))
            check_folder(renderings_root_folder)
            cameras_path = os.path.join(cameras_root_folder,
                "MV_cameras_{}.jpg".format("test"))
            images_path = os.path.join(renderings_root_folder, "MV_renderings_{}.jpg".format("test"))
            auto_render_and_save_images_and_cameras(targets, meshes, extra_info,correction_factor, images_path=images_path,
                                                    cameras_path=cameras_path, models_bag=models_bag, setup=setup, device=None)
    if setup["test_only_retr"]:
        print('\nEvaluation:')
        models_bag["mvnetwork"].eval()
        models_bag["view_selector"].eval()
        models_bag["feature_extractor"].eval()

        # extract features for training (if does not exist yet)
        os.makedirs(os.path.dirname(setup["feature_file"]),exist_ok=True)
        if not os.path.exists(setup["feature_file"]) or not os.path.exists(setup["targets_file"]):
            features, targets = compute_features(train_loader,models_bag, setup)
            np.save(setup["feature_file"],features)
            np.save(setup["targets_file"],targets)

        # reduce Features:
        LFDA_reduction_file = os.path.join(setup["features_dir"], "reduction_LFDA.pkl")
        if not os.path.exists(LFDA_reduction_file):
            from metric_learn import LFDA
            features = np.load(setup["feature_file"])
            targets = np.load(setup["targets_file"])
            lfda = LFDA(n_components=128)
            lfda.fit(features, targets)
            with open(LFDA_reduction_file, "wb") as fobj:
                pkl.dump(lfda, fobj)

        with open(LFDA_reduction_file, "rb") as fobj:
            lfda = pkl.load(fobj)


        avg_test_acc, avg_test_retr_mAP, avg_test_loss, _ = evluate(val_loader, models_bag, setup, retrieval=True)
        print('\tVal Acc: %.2f - val retr-mAP: %.2f - val Loss: %.4f' %
            (avg_test_acc.item(), avg_test_retr_mAP, avg_test_loss.item()))
        print('\tCurrent best val acc: %.2f' % setup["best_acc"])
        
    elif setup["visualize_verts_mode"]:
        from tqdm import tqdm

        list_dict = ListDict(["sample_nb","nb_verts","class_label"])
        print("visualizing train set")
        for ii, (targets, meshes, extra_info,correction_factor) in enumerate(tqdm(dset_train)):
            nb_verts = meshes.verts_padded().shape[1]
            list_dict.append(
                {"sample_nb": ii, "nb_verts": nb_verts, "class_label": targets})
        save_results(os.path.join(setup["verts_dir"],
                                "train_SMPLmesh_verts.csv"), list_dict)

        list_dict = ListDict(["sample_nb", "nb_verts", "class_label"])
        print("visualizing test set")
        for ii, (targets, meshes, extra_info,correction_factor) in enumerate(tqdm(dset_val)):
            nb_verts = meshes.verts_padded().shape[1]
            list_dict.append(
                {"sample_nb": ii, "nb_verts": nb_verts, "class_label": targets})
        save_results(os.path.join(setup["verts_dir"], "test_SMPLmesh_verts.csv"), list_dict)



    if setup["extract_features_mode"]:
        from tqdm import tqdm
        torch.multiprocessing.set_sharing_strategy('file_system')
        if setup["shape_extractor"] == "PointNet":
            point_network = PointNet(40, alignment=True).cuda()
        elif setup["shape_extractor"] == "DGCNN":
            point_network = SimpleDGCNN(40).cuda()
        point_network.eval()
        load_point_ckpt(point_network,  setup,  ckpt_dir='./checkpoint')

        print('\nEvaluation pointnetwork alone:')
        avg_train_acc, avg_train_loss = test_point_network(
            point_network, criterion, data_loader=train_loader, setup=setup, device=None)
        avg_test_acc, avg_loss = test_point_network(
            point_network, criterion, data_loader=val_loader, setup=setup, device=None)
        print('\ttrain acc: %.2f - train Loss: %.4f' %
            (avg_train_acc.item(), avg_train_loss.item()))
        print('\tVal Acc: %.2f - val Loss: %.4f' %
            (avg_test_acc.item(), avg_loss.item()))

        print('\Extracting the point features:')
        train_points_features_list = [file_name.replace(".off", "_PFeautres.pkl") for file_name in dset_train.meshes_list if file_name[-4::] == ".off"]
        test_points_features_list = [file_name.replace(".off", "_PFeautres.pkl") for file_name in dset_val.meshes_list if file_name[-4::] == ".off"]

        for i, (_, _, _, points) in enumerate(tqdm(dset_train)):
            with torch.no_grad():
                points = points[None,...].transpose(1, 2).cuda()
                logits, post_max, transform_matrix = point_network(points)
                saveables = {'logits': logits.cpu().numpy(),
                            'post_max': post_max.cpu().numpy(),
                            "transform_matrix": transform_matrix.cpu().numpy(),
                            }
                save_obj(saveables, train_points_features_list[i])
        print("finished train set")

        for i, (_, _, _, points) in enumerate(tqdm(dset_val)):
            with torch.no_grad():
                points = points[None, ...].transpose(1, 2).cuda()
                logits, post_max, transform_matrix = point_network(points)
                saveables = {'logits': logits.cpu().numpy(),
                            'post_max': post_max.cpu().numpy(),
                            "transform_matrix": transform_matrix.cpu().numpy(),
                            }
                save_obj(saveables, test_points_features_list[i])
        print("finished test set")


    elif setup["late_fusion_mode"]:
        from tqdm import tqdm
        RESNET_FEATURE_SIZE = 40
        models_bag["mvnetwork"].eval()
        models_bag["view_selector"].eval()
        models_bag["feature_extractor"].eval()
        if setup["log_metrics"]:
            writer = SummaryWriter(setup["logs_dir"])
        torch.multiprocessing.set_sharing_strategy('file_system')
        if setup["shape_extractor"] == "PointNet":
            point_network = PointNet(40, alignment=True).cuda()
        elif setup["shape_extractor"] == "DGCNN":
            point_network = SimpleDGCNN(40).cuda()
        point_network.eval()
        load_point_ckpt(point_network,  setup,  ckpt_dir='./checkpoint')

        print('\nEvaluation pointnetwork alone:')
        # all_imgs_list = [640,669,731,2100,2000]
        # all_imgs_list = [2529]

        # all_imgs_list = [2438,  2439,  520,  2573,  527,  2448,  2447,  2449,  2575,  534,  152,  2586,  2458,  2464,  39,  2472,  2425,  426,427,              431,                     47,                     51,      2487,      2489,      58,      2492,      2493,      450,      579,      68,      2501,      73,      2514,      212,      469,      2525,                     94,                     93,                     2529,                     2535,                     2536,                     487,                     234,                     2539,                     237,                     2418,                     505]
        all_imgs_list = [118, 119, 120, 251, 260, 269, 323, 355, 468, 479, 607, 620, 673, 711, 713, 715, 723, 740, 751, 759, 782, 783, 788, 791, 800, 812, 816, 856, 878, 882, 886, 888, 891, 908, 926, 927, 942, 943, 944, 945, 946, 947, 948, 951, 953, 955, 956, 957, 958, 960, 961, 972, 1033, 1057, 1186, 1187, 1195, 1270, 1299, 1303, 1344, 1424, 1428, 1444, 1457, 1468, 1473, 1475, 1478, 1498,
                        1499, 1504, 1506, 1532, 1550, 1568, 1587, 1594, 1633, 1645, 1648, 1651, 1655, 1677, 1712, 1746, 1749, 1776, 1821, 1853, 1859, 1867, 1868, 1869, 1923, 1988, 1993, 1996, 2000, 2001, 2026, 2042, 2050, 2053, 2065, 2089, 2114, 2115, 2295, 2312, 2314, 2322, 2330, 2332, 2335, 2336, 2337, 2342, 2360, 2375, 2380, 2381, 2382, 2386, 2389, 2391, 2395, 2400, 2405, 2409, 2416, 2420, 2429, 2441]
        # all_imgs_list = list(range(len(dset_val)))
        # all_imgs_list = [9,17,24,27,31,65,70,94,102,109,112,114,115,132,139,141,143,144,147,148,176,223,228,252,257,259,271,274,276,287,289,290,295,298,301,306,310,312,315,317,318,319,322,330,332,340,346,347,348,353,355,356,357,358,359,364,365,367,369,378,380,384,386,387,391,393,423,436,438,455,456,464,465,467,471,476,477,479,480,481,483,486,488,490,492,493,494,496,499,500,501,506,511,515,516,517,519,520,522,523,524,530,534,535,536,538,540,543,545,546,550,558,561,562,565,570,571,572,573,574,578,580,581,584,585,587,593,595,596,611,612,617,630,637,638,643,644,649,655,684,686,693,694]
        visualize_retrieval_views(dset_val, all_imgs_list,
                                models_bag, setup, device)
        # compiled_analysis_list =  analyze_rendered_views(dset_val, all_imgs_list,models_bag, setup, device)
        # f = open('test_avg_pos.txt', 'w')
        # f.writelines(["{:.3f} \n".format(x) for x in compiled_analysis_list])
        # f.close()
        # avg_train_acc, avg_train_loss = test_point_network(
        #     point_network, criterion, data_loader=train_loader, setup=setup, device=None)
        # avg_test_acc, avg_loss = test_point_network(
        #     point_network, criterion, data_loader=val_loader, setup=setup, device=None)
        # print('\ttrain acc: %.2f - train Loss: %.4f' %
        #       (avg_train_acc.item(), avg_train_loss.item()))
        # print('\tVal Acc: %.2f - val Loss: %.4f' %
        #       (avg_test_acc.item(), avg_loss.item()))
        raise Exception("just checking the visualization")
        print('\Training late fusuion of pointnetwork and MV network:')
        classifier = Seq(MLP([setup["features_size"]+RESNET_FEATURE_SIZE, setup["features_size"], setup["features_size"], 5 * setup["nb_views"],
                            2*setup["nb_views"]], dropout=0.5, norm=True), MLP([2*setup["nb_views"], 40], act=None, dropout=0, norm=False), )
        cls_optimizer = torch.optim.AdamW(
            classifier.parameters(), lr=lr, weight_decay=setup["weight_decay"])
        fe_optimizer = torch.optim.AdamW(
            point_network.parameters(), lr=lr,)
        models_bag["classifier"] = classifier.cuda()
        models_bag["cls_optimizer"] = cls_optimizer
        models_bag["fe_optimizer"] = fe_optimizer
        models_bag["point_network"] = point_network
        for epoch in range(setup["start_epoch"], n_epochs):
            setup["c_epoch"] = epoch
            print('\n-----------------------------------')
            print('Epoch: [%d/%d]' % (epoch+1, n_epochs))
            start = time.time()
            models_bag["mvnetwork"].train()
            models_bag["view_selector"].train()
            models_bag["feature_extractor"].train()
            models_bag["classifier"].train()
            models_bag["point_network"].train()

            avg_train_acc, avg_train_loss = train_late_fusion(
                train_loader, models_bag, setup)
            print('Time taken: %.2f sec.' % (time.time() - start))

            models_bag["mvnetwork"].eval()
            models_bag["view_selector"].eval()
            models_bag["feature_extractor"].eval()
            models_bag["classifier"].eval()
            models_bag["point_network"].eval()
            avg_test_acc, avg_loss, views_record = evluate_late_fusion(
                val_loader, models_bag, setup)

            print('\nEvaluation:')
            print('\ttrain acc: %.2f - train Loss: %.4f' %
                (avg_train_acc.item(), avg_train_loss.item()))
            print('\tVal Acc: %.2f - val Loss: %.4f' %
                (avg_test_acc.item(), avg_loss.item()))
            print('\tCurrent best val acc: %.2f' % setup["best_acc"])
            if setup["log_metrics"]:
                writer.add_scalar('Loss/train', avg_train_loss.item(), epoch)
                writer.add_scalar('Loss/val', avg_loss.item(), epoch)
                writer.add_scalar('Accuracy/train', avg_train_acc.item(), epoch)
                writer.add_scalar('Accuracy/val', avg_test_acc.item(), epoch)

            # Log epoch to tensorboard
            # See log using: tensorboard --logdir='logs' --port=6006 ######################################
            # util.logEpoch(logger, mvnetwork, epoch + 1, avg_loss, avg_test_acc) #############################################
            saveables = {'epoch': epoch + 1,
                        'state_dict': models_bag["mvnetwork"].state_dict(),
                        # "view_selector": models_bag["view_selector"].state_dict(),
                        # "feature_extractor": models_bag["feature_extractor"].state_dict(),
                        "classifier": models_bag["classifier"].state_dict(),
                        "cls_optimizer": models_bag["cls_optimizer"].state_dict(),
                        "fe_optimizer": models_bag["fe_optimizer"].state_dict(),
                        "point_network": models_bag["point_network"].state_dict(),
                        'acc': avg_test_acc,
                        'best_acc': setup["best_acc"],
                        'optimizer': models_bag["optimizer"].state_dict(),

                        # 'vs_optimizer': None if not setup["is_learning_views"] else models_bag["vs_optimizer"].state_dict(),
                        # 'fe_optimizer': None if not setup["is_learning_points"] else models_bag["fe_optimizer"].state_dict(),
                        }
            if setup["save_all"]:
                save_checkpoint(saveables, setup, views_record,
                                setup["weights_file"])
            # Save mvnetwork
            if avg_test_acc.item() >= setup["best_acc"]:
                print('\tSaving checkpoint - Acc: %.2f' % avg_test_acc)
                saveables["best_acc"] = avg_test_acc
                setup["best_loss"] = avg_loss.item()
                setup["best_acc"] = avg_test_acc.item()
                save_checkpoint(saveables, setup, views_record,
                                setup["weights_file"])

            # Decaying Learning Rate
            if (epoch + 1) % setup["lr_decay_freq"] == 0:
                lr *= setup["lr_decay"]
                models_bag["optimizer"] = torch.optim.AdamW(
                    models_bag["mvnetwork"].parameters(), lr=lr)
                print('Learning rate:', lr)



    elif setup["custom_views_mode"]:
        print('\nEvaluation on custom views:')
        models_bag["mvnetwork"].eval()
        models_bag["view_selector"].eval()
        models_bag["feature_extractor"].eval()
        azim_dict = {0: [-192.66859436035156, -15.20999135017395], 1: [-191.53206451416017, -18.21193141102791], 2: [-189.87418563842775, -22.727777905464173], 3: [-189.5013671875, -23.754320549964905], 4: [-188.9880499267578, -25.210509920120238], 5: [-188.87568756103516, -25.400040702819823], 6: [-189.2273208618164, -24.596572971343996], 7: [-191.38295715332032, -18.638404846191406], 8: [-189.22774841308595, -24.548475017547606], 9: [-189.11621627807617, -24.74792947769165], 10: [-189.89229507446288, -22.889934682846068], 11: [-189.6816078186035, -23.35349311828613], 12: [-190.6269179942996, -20.706220472967903], 13: [-188.54616317749023, -26.346070098876954], 14: [-188.61920787012855, -26.109867694766024], 15: [-189.05272064208984, -24.96319999694824], 16: [-189.29007568359376, -24.259252433776854], 17: [-189.67282363891601, -23.466498575210572], 18: [-191.72444534301758, -18.160002851486205], 19: [-189.42255096435548, -23.99682159423828],
                    20: [-189.23582611083984, -24.50362663269043], 21: [-189.2811750793457, -24.43984027862549], 22: [-190.2920558166504, -21.671046514511108], 23: [-189.65490580714027, -23.397453385730124], 24: [-188.95428543090821, -25.283514785766602], 25: [-191.02093780517578, -19.77732984304428], 26: [-189.86249114990235, -22.906716599464417], 27: [-189.91762008666993, -22.795362377166747], 28: [-188.94511810302734, -25.321912517547606], 29: [-190.33770828247071, -21.585721135139465], 30: [-190.51060180664064, -20.926687784194947], 31: [-190.59102325439454, -20.871350955963134], 32: [-188.79449234008788, -25.710672760009764], 33: [-192.12433547973632, -16.589099179506302], 34: [-190.01130752563478, -22.284782123565673], 35: [-190.927488861084, -20.16731552839279], 36: [-189.3496792602539, -24.16118221282959], 37: [-189.8223486328125, -22.968083379268645], 38: [-189.33740615844727, -24.32109651565552], 39: [-189.86952896118163, -22.876914024353027]}

        elev_dict = {0: [-4.426920528411865, 59.01743507385254], 1: [-2.883237533569336, 56.864502029418944], 2: [-0.13723064422607428, 54.486220321655274], 3: [0.5127001762390135, 54.005449295043945], 4: [1.6092494773864745, 53.62236198425293], 5: [1.3898611450195313, 52.847849197387696], 6: [1.4608168601989742, 54.3101411819458], 7: [-2.9375133895874024, 56.301391677856444], 8: [1.2852876091003418, 54.0532292175293], 9: [0.8941428184509276, 53.0734525680542], 10: [0.7435904502868651, 55.83231945037842], 11: [0.38383121490478506, 54.55170783996582], 12: [-1.2894535508266716, 55.70144302900447], 13: [2.195459079742432, 52.75041961669922], 14: [1.975170490353606, 52.698735303657], 15: [1.438679313659668, 53.57254428863526], 16: [0.5812169265747069, 53.28563362121582], 17: [0.8869257736206054, 55.228461837768556], 18: [-1.38173246383667, 59.816728019714354], 19: [0.8406832695007322, 54.16661205291748], 20: [1.070935535430908, 53.78573017120361],
                    21: [1.2648774337768556, 54.246552734375], 22: [-0.9033590698242188, 55.075473403930665], 23: [0.7030576439790946, 54.83112379562023], 24: [1.6217111587524413, 53.50487461090088], 25: [-1.2097094535827637, 57.31082511901855], 26: [0.49821533203124985, 55.36620071411133], 27: [0.3556199073791504, 55.41757221221924], 28: [1.738970775604248, 53.63179256439209], 29: [-0.20831499099731454, 56.12408332824707], 30: [-1.9850734329223634, 54.34881855010986], 31: [-0.8675153732299808, 56.16395282745361], 32: [1.9273006439208982, 53.32044620513916], 33: [-3.696138343811035, 57.90461875915528], 34: [-0.6039499282836917, 54.31502857208252], 35: [-0.5853189849853515, 57.87767822265625], 36: [0.7845332717895508, 53.807646102905274], 37: [0.46777093887329085, 55.14418148040772], 38: [1.2266029357910155, 54.42551383972168], 39: [0.17910842895507817, 54.98482704162598]}
        
        models_bag["azim_dict"] = azim_dict
        models_bag["elev_dict"] = elev_dict
        avg_test_acc, avg_test_loss, _ = evluate(val_loader, models_bag, setup)
        print('\tVal Acc: %.2f - val Loss: %.4f' %
            (avg_test_acc.item(), avg_test_loss.item()))
        print('\tCurrent best val acc: %.2f' % setup["best_acc"])




    elif setup["occlusion_robustness_mode"]:
        models_bag["mvnetwork"].eval()
        models_bag["view_selector"].eval()
        models_bag["feature_extractor"].eval()
        if "modelnet" not in setup["mesh_data"].lower():
            raise Exception('Occlusion is only supported froom ModelNet now ')
        from tqdm import tqdm
        torch.multiprocessing.set_sharing_strategy('file_system')


        print('\Evaluatiing om the cropped data :')

        override = True
        networks_list = ["MVTN"]
        # networks_list = ["MVCNN","PointNet", "DGCNN"]
        # network = "MVCNN"
        # out_dir = "occlusion_data"
        # angle = setup["initial_angle"]
        # rot_axis = [1, 0, 0]
        # number_of_exps = 41
        # target_indx = 5
        # targets_list = [0, 5, 35, 2, 8, 33, 22, 37, 4, 30]
        # victims_list = [0, 5, 35, 2, 8, 33, 22, 37, 4, 30]
        # # victims_list = [0]
        # factor_list = list(np.linspace(-1, 1, num=number_of_exps))
        factor_list = [-0.75,-0.5,-0.3,-0.2,-0.1,0,0.1,0.2,0.3,0.5,0.75]
        axis_list = [0, 1, 2]
        # axis_list = [1]

        setup_keys = ["network", "batch", "factor", "axis"]
        setups = ListDict(setup_keys)
        results = ListDict(["prediction", "class"])
        for network in networks_list: 
            if network == "PointNet":
                setup["shape_extractor"] = "PointNet"
                point_network = PointNet(40, alignment=True).cuda()
            elif network == "DGCNN":
                setup["shape_extractor"] = "DGCNN"
                point_network = SimpleDGCNN(40).cuda()
            if network in ["DGCNN","PointNet"]:
                point_network.eval()
                load_point_ckpt(point_network,  setup,  ckpt_dir='./checkpoint')
            exp_id = "chopping_{}".format(network)
            save_file = os.path.join(setup["results_dir"], exp_id+".csv")
            if not os.path.isfile(save_file) or override:
                t = tqdm(enumerate(val_loader), total=len(val_loader))
                for ii, (targets, meshes, orig_pts, correction_factor) in t:
                    c_batch = len(meshes)
                    with torch.no_grad():
                        rendered_images, _, azim, elev, dist = auto_render_meshes(
                            targets, meshes, orig_pts, correction_factor, models_bag, setup, device=None)
                        targets = targets.cuda()
                        for factor in factor_list:
                            for axis in axis_list:
                                c_setup = {"network": network, 
                                        "batch": ii, "factor": factor, "axis": axis}
                                [setups.append(c_setup) for ii in range(c_batch)]
                                chopped_pts = chop_ptc(orig_pts.cpu().numpy(), factor, axis=axis)
                                chopped_pts = torch.from_numpy(chopped_pts)
                                if network not in ["PointNet","DGCNN"]:
                                    rendered_images, _, _, _, _ = auto_render_meshes(
                                        targets, chopped_pts, chopped_pts, 1.0, models_bag, setup, device=None)
                                    # save_grid(image_batch=rendered_images[0, ...],save_path=os.path.join(setup["results_dir"],"renderings","{}_{}.jpg".format(network,factor)), nrow=setup["nb_views"])
                                    outputs, _ = models_bag["mvnetwork"](rendered_images)
                                else: 
                                    chopped_pts = chopped_pts.transpose(1, 2).cuda()
                                    outputs = point_network(chopped_pts)[0].view(c_batch, -1)
                                _, predictions = torch.max(outputs.data, 1)
                                c_result = ListDict({"prediction": predictions.cpu().numpy(
                                ).tolist(), "class": targets.cpu().numpy().tolist()})
                                results.extend(c_result)
                                save_results(save_file, results+setups)
                            # raise Exception("just checking the visualization")


    elif setup["rotation_robustness_mode"]:
        setup["results_file"] = os.path.join(setup["results_dir"], setup["exp_id"]+"_robustness_{}.csv".format(str(int(setup["max_degs"]))))
        setup["return_points_saved"] = True
        assert os.path.isfile(setup["weights_file"]
                            ), 'Error: no checkpoint file found!'

        loaded_info = load_results(os.path.join(
            setup["results_dir"], setup["exp_id"]+"_accuracy.csv"))
        setup["start_epoch"] = loaded_info["start_epoch"][0]
        setup["nb_views"] = loaded_info["nb_views"][0]
        setup["selection_type"] = loaded_info["selection_type"][0]

        # exp_ids_list = [setup["exp_id"]]
        print('\nEvaluating Robustness:')
        # for exp_id in exp_ids_list:
        # setup["exp_id"] = exp_id
        view_selector = ViewSelector(setup["nb_views"], selection_type=setup["selection_type"],
                                    canonical_elevation=setup["canonical_elevation"], canonical_distance=setup["canonical_distance"],
                                    shape_features_size=setup["features_size"], transform_distance=setup["transform_distance"], input_view_noise=setup["input_view_noise"], light_direction=setup["light_direction"]).cuda()
        feature_extractor = FeatureExtracter(setup).cuda()
        models_bag["view_selector"] = view_selector
        models_bag["feature_extractor"] = feature_extractor
        load_checkpoint_robustness(setup, models_bag, setup["weights_file"])
        models_bag["mvnetwork"].eval()
        models_bag["view_selector"].eval()
        models_bag["feature_extractor"].eval()
        acc_list = []
        for _ in range(setup["repeat_exp"]):
            avg_test_acc, _ = evluate_rotation_robustness(
                val_loader, models_bag, setup, max_degs=setup["max_degs"])
            acc_list.append(avg_test_acc.item())
        setup["best_acc"] = np.mean(acc_list)
        print("exp: {} \tVal Acc: {:.2f} ".format(
            setup["exp_id"], setup["best_acc"]))
        setup_dict = ListDict(list(setup.keys()))
        save_results(setup["results_file"], setup_dict.append(setup))




elif setup["mvnetwork"] == "rotnet":
    if setup["batch_size"] % setup["nb_views"] != 0:
        raise ValueError("batch size should be multiplication of the number of views")
    vcand = np.load('rotationNet/vcand_case1.npy')
    if setup["log_metrics"]: 
        writer = SummaryWriter(setup["logs_dir"])
    for epoch in range(setup["start_epoch"], n_epochs):
        setup["c_epoch"] = epoch
        print('\n-----------------------------------')
        print('Epoch: [%d/%d]' % (epoch+1, n_epochs))
        if setup["train"]:
            start = time.time()
            models_bag["mvnetwork"].train()
            models_bag["view_selector"].train()
            models_bag["feature_extractor"].train()
            avg_train_acc, avg_train_loss = train_rotationNet(train_loader, models_bag, setup)
            print('Time taken: %.2f sec.' % (time.time() - start))
            print('\ttrain acc: %.2f - train Loss: %.4f' %
                (avg_train_acc, avg_train_loss.item()))
            models_bag["mvnetwork"].eval()
            models_bag["view_selector"].eval()
            models_bag["feature_extractor"].eval()
            avg_test_acc, avg_loss = evaluate_rotationNet(
                val_loader, models_bag, setup)

        print('\nEvaluation:')

        print('\tVal Acc: %.2f - val Loss: %.4f' %
              (avg_test_acc, avg_loss.item()))
        print('\tCurrent best val acc: %.2f' % setup["best_acc"])
        if setup["log_metrics"] and setup["train"]:
            writer.add_scalar('Loss/train', avg_train_loss.item(), epoch)
            writer.add_scalar('Loss/val', avg_loss.item(), epoch)
            writer.add_scalar('Accuracy/train', avg_train_acc, epoch)
            writer.add_scalar('Accuracy/val', avg_test_acc, epoch)

        saveables = {'epoch': epoch + 1,
                     'state_dict': models_bag["mvnetwork"].state_dict(),
                     "view_selector": models_bag["view_selector"].state_dict(),
                     "feature_extractor": models_bag["feature_extractor"].state_dict(),
                     'acc': avg_test_acc,
                     'best_acc': setup["best_acc"],
                     'optimizer': models_bag["optimizer"].state_dict(),
                     'vs_optimizer': None if not setup["is_learning_views"] else models_bag["vs_optimizer"].state_dict(),
                     'fe_optimizer': None if not setup["is_learning_points"] else models_bag["fe_optimizer"].state_dict(),
                     }

        # Save mvnetwork
        if avg_test_acc >= setup["best_acc"]:
            print('\tSaving checkpoint - Acc: %.2f' % avg_test_acc)
            saveables["best_acc"] = avg_test_acc
            setup["best_loss"] = avg_loss.item()
            setup["best_acc"] = avg_test_acc
            save_checkpoint(saveables, setup, None,
                            setup["weights_file"])

        # Decaying Learning Rate
        if (epoch + 1) % setup["lr_decay_freq"] == 0:
            lr *= setup["lr_decay"]
            models_bag["optimizer"] = torch.optim.AdamW(
                models_bag["mvnetwork"].parameters(), lr=lr)
            print('Learning rate:', lr)
    if setup["log_metrics"] and setup["train"]:
        writer.add_hparams(setup, {"hparams/best_acc": setup["best_acc"]})

elif setup["mvnetwork"] == "viewgcn":
    if setup["resume_mvtn"]:
        load_mvtn(setup, models_bag, setup["weights_file2"])
        setup["vs_learning_rate"] = 0.0
        setup["pn_learning_rate"] = 0.0

    def view_gcn_exp(setup, models_bag, train_loader, val_loader, dset_val):
        seed_torch()
        # STAGE 1


        models_bag["mvnetwork"].train()
        models_bag["view_selector"].train()
        models_bag["feature_extractor"].train()

        trainer = ModelNetTrainer_mvt(models_bag, train_loader, val_loader, dset_val, nn.CrossEntropyLoss(
        ), 'svcnn', setup["checkpoint_dir1"], num_views=1, setup=setup, classes=classes)


        if setup["resume_first"]:
            trainer.model.load(trainer.weights_dir,)
        if setup["phase"] == "all" or setup["phase"] == "first":
            if not setup["test_only"]:
                trainer.train(setup["first_stage_epochs"])
            else:
                trainer.visualize_views("test", [55, 66, 77])
                trainer.update_validation_accuracy(1)

        # # # STAGE 2
        models_bag["mvnetwork"] = view_GCN(setup["exp_id"], models_bag["mvnetwork"], nclasses=len(classes),
                                        cnn_name=setup["cnn_name"], num_views=setup["nb_views"])
        models_bag["optimizer"] = torch.optim.SGD(models_bag["mvnetwork"].parameters(), lr=setup["learning_rate"],
                                            weight_decay=setup["weight_decay"], momentum=0.9)

        trainer = ModelNetTrainer_mvt(models_bag, train_loader, val_loader, dset_val,
                                    nn.CrossEntropyLoss(), 'view-gcn', setup["checkpoint_dir2"], num_views=setup["nb_views"], setup=setup, classes=classes)

        if setup["resume_second"] or setup["test_only"]:
            trainer.model.load(trainer.weights_dir,)
            if setup["is_learning_views"]:
                load_mvtn(setup, models_bag, setup["weights_file2"])
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
    view_gcn_exp(setup, models_bag,train_loader, val_loader, dset_val)
if setup["measure_speed_mode"]:

    models_bag["mvnetwork"].eval()
    models_bag["view_selector"].eval()
    models_bag["feature_extractor"].eval()
    if "modelnet" not in setup["mesh_data"].lower():
        raise Exception('Occlusion is only supported froom ModelNet now ')
    from tqdm import tqdm
    torch.multiprocessing.set_sharing_strategy('file_system')

    print('\Evaluatiing speed memory  :')
    print("network", "\t", "MACs", "\t", "# params", "\t", "time/sample (ms)")
    override = True
    MAX_ITER = 10000
    for network in ["MVTN", "PointNet", "DGCNN", "MVCNN"]:
        if network == "PointNet":
            setup["shape_extractor"] = "PointNet"
            point_network = PointNet(40, alignment=True).cuda()
        elif network == "DGCNN":
            setup["shape_extractor"] = "DGCNN"
            point_network = SimpleDGCNN(40).cuda()
        if network in ["DGCNN", "PointNet"]:
            point_network.eval()
            load_point_ckpt(point_network,  setup,
                            ckpt_dir='./checkpoint', verbose=False)
            macs, params = get_model_complexity_info(
                point_network, (3, setup["nb_points"]), as_strings=True, print_per_layer_stat=False, verbose=False)
            inp = torch.rand((1, 3, setup["nb_points"])).cuda()
            avg_time = profile_op(MAX_ITER, point_network, inp)
        elif network in ["MVCNN", "ViewGCN"]:
            macs, params = get_model_complexity_info(
                models_bag["mvnetwork"], (setup["nb_views"], 3, setup["image_size"], setup["image_size"]), as_strings=True, print_per_layer_stat=False, verbose=False)
            inp = torch.rand(
                (1, setup["nb_views"], 3, setup["image_size"], setup["image_size"])).cuda()
            avg_time = profile_op(MAX_ITER, models_bag["mvnetwork"], inp)
        else:
            macs, params = get_model_complexity_info(
                models_bag["view_selector"], (setup["features_size"],), as_strings=False, print_per_layer_stat=False, verbose=False)
            inp = torch.rand((1, setup["features_size"])).cuda()
            avg_time = profile_op(MAX_ITER, models_bag["view_selector"], inp)
        print(network, "\t", macs, "\t", params,
              "\t", "{}".format(avg_time*1e3))
