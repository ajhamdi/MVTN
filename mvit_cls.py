import torch
from torchvision.models import segmentation
import timm
import torch.nn as nn
import sys
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.autograd import Variable
from einops import rearrange
import warnings
import scipy.spatial		
from tqdm import tqdm	
import pickle as pkl
from vit.vit_pytorch.vit import ViT, MViT
import torchvision
# https://htmlcolorcodes.com/color-names/
from colors import color2rgb, color2rgba


import torchvision.transforms as transforms

import argparse
import numpy as np
import time
import os

# from models.mvcnn import *
from util import *
from ops import *
from seg_ops import * 


# from logger import Logger
from torch.utils.tensorboard import SummaryWriter
from custom_dataset import MultiViewDataSet, ThreeMultiViewDataSet, collate_fn, ShapeNetCore, ScanObjectNN, PartNormalDataset  # , ModelNet40


PLOT_SAMPLE_NBS = [222,357, 1402, 1984, 1057, 2201, 1355, 1875]

# colors used in visualization of the segmentation 
color_label_list = ["red", "green", "blue", "brown", "purple","orange", "yellow", "black", "red3", "green3", "blue3"]
COLOR_LABEL_VALUES_LIST = [color2rgb(clr) for clr in color_label_list]


parser = argparse.ArgumentParser(description='MVCNN-PyTorch')
parser.add_argument('--depth', type=int,  default=18, help='resnet depth (default: resnet18)')
parser.add_argument('--gpu', type=int,
                     default=0, help='GPU number ')
parser.add_argument('--mvnetwork', '-m',  default='resnet', choices=['resnet', 'mvcnn', "vit", "mvit", "wvit"],
                    help='pretrained mvnetwork: ' + ' | '.join(['resnet', 'mvcnn', "vit", "mvit"]) + ' (default: {})'.format('resnet'))
parser.add_argument('--epochs', default=100, type=int,  help='number of total epochs to run (default: 100)')
parser.add_argument('-b', '--batch-size', default=20, type=int,
                     help='mini-batch size (default: 4)')

parser.add_argument('--image_data',required=False,  help='path to 2D dataset')
parser.add_argument('--mesh_data', required=True,  help='path to 3D dataset')
parser.add_argument('--exp_set', type=str, default='00', help='pick ')
parser.add_argument('--exp_id', type=str, default='random', help='pick ')
parser.add_argument('--nb_views', default=4, type=int, 
                    help='number of views in MV CNN')
parser.add_argument('--image_size', default=224, type=int, 
                    help='the size of the images rendered by the differntibe renderer ( other poissible 384)')
parser.add_argument('--canonical_elevation', default=30.0, type=float,
                     help='if selection_type== canoncal , the elevation of the view points is givene by this angle')
parser.add_argument('--canonical_distance', default=2.2, type=float,
                     help='the distnace of the view points from the center if the object  ')
parser.add_argument('--selection_type', '-s',  default="circular", choices=["circular", "random", "learned_offset", "learned_direct", "spherical", "learned_spherical", "learned_random", "learned_transfer", "custom"],
                    help='the selection type of views ')
parser.add_argument('--plot_freq', default=3, type=int, 
                    help='the frequqency of plotting the renderings and camera positions')
parser.add_argument('--renderings_dir', '-rd',  default="renderings",help='the destinatiojn for the renderings ')
parser.add_argument('--results_dir', '-rsd',  default="mvit_results",help='the destinatiojn for the results ')
parser.add_argument('--logs_dir', '-lsd',  default="logs",help='the destinatiojn for the tensorboard logs ')
parser.add_argument('--cameras_dir', '-c', 
                    default="cameras", help='the destination for the 3D plots of the cameras ')
parser.add_argument('--simplified_mesh', dest='simplified_mesh',
                    action='store_true', help='use simplified meshes in learning .. not the full meshes ... it ends in `_SMPLER.obj` ')
parser.add_argument('--cleaned_mesh', dest='cleaned_mesh',
                    action='store_true', help='use cleaned meshes using reversion of light direction for faulted meshes')


parser.add_argument('--lr', '--learning_rate', default=0.00001, type=float,
                     help='initial learning rate (default: 0.0001)')
parser.add_argument('--weight_decay', default=0.3, type=float,
                    help='weight decay for MVT ... default 0.01')
parser.add_argument('--momentum', default=0.9, type=float, 
                    help='momentum (default: 0.9)')
parser.add_argument('--lr-decay-freq', default=25, type=float,
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
parser.add_argument('--ignore_saving_models', dest='ignore_saving_models',
                    action='store_true', help='DO NOT save the best model checkpoint and results .... default saves only best test accuracy epoch')
parser.add_argument('--test_only', dest='test_only',
                    action='store_true', help='do only testing once ... no training ')
parser.add_argument('--train_only', dest='train_only',
                    action='store_true', help='do only training  once ... no testing ')
parser.add_argument('--test_retrieval_only', dest='test_retrieval_only',
                    action='store_true', help='do only testing once including metrics for retrieval ... no training ')
parser.add_argument('--extract_features_mode', dest='extract_features_mode',
                    action='store_true', help='extract shape features from the 3d models to be used in training  the view selector')
parser.add_argument('--visualize_verts_mode', dest='visualize_verts_mode',
                    action='store_true', help='do visualization ... no evaluations ')
parser.add_argument('--return_points_sampled', dest='return_points_sampled',
                    action='store_true', help='reuturn 3d point clouds from the data loader sampled from hte mesh ')
parser.add_argument('--return_points_saved', dest='return_points_saved',
                    action='store_true', help='reuturn 3d point clouds from the data loader saved under `filePOINTS.pkl` ')
parser.add_argument('--return_extracted_features', dest='return_extracted_features',
                    action='store_true', help='return pre extracted features `*_PFeatures.pt` for each 3d model from the dataloader ')
parser.add_argument('--custom_views_mode', dest='custom_views_mode',
                    action='store_true', help=' test MVCNN with `custom` views ')
parser.add_argument('--features_type', '-ftpe',  default="post_max", choices=["logits", "post_max", "transform_matrix",                                                                              "pre_linear", "logits_trans", "post_max_trans", "pre_linear_trans"],help='the type of the features extracted from the feature extractor ( early , middle , late) ')
parser.add_argument('--log_metrics', dest='log_metrics',
                    action='store_true', help='logs loss and acuracy and other metrics in `logs_dir` for tensorboard ')
parser.add_argument('--light_direction', '-ldrct',  default="random", choices=["fixed", "random", "relative"],
                    help='apply random light direction on the rendered images .. otherwise default (0, 1.0, 0)')
parser.add_argument('--cull_backfaces', dest='cull_backfaces',
                    action='store_true', help='cull back_faces ( remove them from the image) ')
parser.add_argument('--view_reg', default=0.0, type=float,
                    help='use regulizer to the learned view selector so they can be apart ...ONLY when `selection_type` == learned_direct   (default: 0.0)')

## point cloud rnedienring 
parser.add_argument('--pc_rendering', dest='pc_rendering',
                    action='store_true', help='use point cloud renderer instead of mesh renderer  ')
parser.add_argument('--points_radius', default=0.006, type=float,
                    help='the size of the rendered points if `pc_rendering` is True  ')
parser.add_argument('--points_per_pixel',  default=1, type=int,
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
parser.add_argument('--augment_training', dest='augment_training',
                    action='store_true', help='augment the training of the CNN by scaling , rotation , translation , etc ')
parser.add_argument('--crop_ratio', default=0.3, type=float,
                    help='the crop ratio of the images when `augment_training` == True  ')  

#### MV Transfoemer       
parser.add_argument('--patch_size', default=16, type=int,help='patch size of the MVIT ')
parser.add_argument('--mvit_dim', default=768, type=int,help='te dimension of the multi-view transfomerer features')
parser.add_argument('--mlp_dim', default=2048, type=int,help='te dimension of the MLP for classification')
parser.add_argument('--mvit_heads', default=16, type=int,help='te numbner of heads in multihead self-attnetion in the mvit')
parser.add_argument('--mvit_dropout', default=0.1, type=float,help='the dropit at the main mvit  ')
parser.add_argument('--emb_dropout', default=0.1, type=float,help='the dropit at the embedding of the mvit  ')
parser.add_argument('--mv_agr_type', '-mvaggr',  default="max",
                    choices=["max", "mean"], help='pool type of the multi-view setup')
parser.add_argument('--vit_agr_type', '-vitaggr',  default="cls",
                    choices=["mean", "cls"], help='pool type must be either cls (cls token) or mean (mean pooling)')
parser.add_argument('--nb_windows', default=1, type=int,
                    help='the number of windows if `mvnetwork` == `wvit`. if it is 1 it collapses to mvit , if it is = nb_views, it collapses to vit ')
parser.add_argument('--vit_variant',  default="vit",
                    choices=["vit", "swin", "vit_deit"], help='the type of the vision transformer used: vanillar vit or swin or ...')
parser.add_argument('--vit_model_size',  default="base",
                    choices=["base", "small", "tiny","large","huge"], help='the type of the vision transformer used: vanillar vit or swin or ...')
parser.add_argument('--swin_window_size', default=7, type=int,
                    help='window size of Swin transformer (if `vit_variant` == `swin`) ')
parser.add_argument('--pretrained_21k', dest='pretrained_21k',
                    action='store_true', help='use the pretrained weights on ImageNet 22K if available , else the regular 1K imageNEt  ')

# part Segmentation 
parser.add_argument('--part_seg_mode', dest='part_seg_mode',
                    action='store_true', help=' run the MV models for part segmentation')
parser.add_argument('--test_part_seg_only', dest='test_part_seg_only',
                    action='store_true', help='only perform testing and visualization of trained models on part segmentation in `exp_id`')
parser.add_argument('--post_process_iters', default=1, type=int,
                    help='the number of post processing iterations with nearest neighbor propogation , if 0 : no post processing in evaluation ')
parser.add_argument('--post_process_k', default=10, type=int,
                    help='the number of K neightbor used in post processing grows with power of k every iteration  if 0 : no post processing in evaluation ')
parser.add_argument('--parallel_head', dest='parallel_head',
                    action='store_true', help='do segmntation as parallel  heads whwere each head is focused on one class ')
parser.add_argument('--lifting_method',  default="mode",
                    choices=["mode", "mlp", "mean","view_attention","pixel_attention","point_attention"], help='the type of operation used to lift the 2d predictions to 3d predictions')
parser.add_argument('--record_extra_metrics', dest='record_extra_metrics',
                    action='store_true', help='record_extra_metrics like the percentage of every class in points/pixels to its IOU')
parser.add_argument('--add_loss_factor', dest='add_loss_factor',
                    action='store_true', help='do focal loss on the part segmentation based on the class frequency')
parser.add_argument('--use_normals', dest='use_normals',
                    action='store_true', help='use normals as colors for point renderings')
parser.add_argument('--color_normal_p',  default="inf",
                    choices=["inf", "2"], help='the `p` value of the norm that is used to normalize the color of the points by the normals')
parser.add_argument('--transform_distance', dest='transform_distance',
                    action='store_true', help='use rnadomized distance to the object')


# scene segmentation 
parser.add_argument('--scene_seg_mode', dest='scene_seg_mode',
                    action='store_true', help=' run the MV models for scene segmentation')

def model_name_from_setup(setup):
    imagenet_ptrain = "" if not setup["pretrained_21k"] else "_in21k"
    swin_window = "" if "swin" not in setup["vit_variant"] else "_window{}".format(setup["swin_window_size"])
    timm_model_name = "{}_{}_patch{}{}_{}{}".format(setup["vit_variant"], setup["vit_model_size"], setup["patch_size"],swin_window, setup["image_size"], imagenet_ptrain)
    return timm_model_name

args = parser.parse_args()
setup = vars(args)
initialize_setup(setup)

print('Loading data')

transform = None
# a function to preprocess pytorch3d Mesh onject

# device = torch.device("cuda:{}".format(str(setup["gpu"])) if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(int(setup["gpu"]))
if "modelnet" in setup["mesh_data"].lower():
    dset_train = ThreeMultiViewDataSet(
        'train', setup, transform=transform, is_rotated=False)
    dset_val = ThreeMultiViewDataSet(
    'test', setup, transform=transform, is_rotated=False)
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
elif "part" in setup["mesh_data"].lower():
    dset_train = PartNormalDataset(root=setup["mesh_data"], npoints=setup["nb_points"], split='trainval', class_choice=None, normal_channel=setup["use_normals"])
    dset_val = PartNormalDataset(root=setup["mesh_data"], npoints=setup["nb_points"],                                 split='test', class_choice=None, normal_channel=setup["use_normals"])
    parts_per_class = dset_train.parts_per_class
    classes = sorted(list(dset_train.seg_classes.keys()))

train_loader = DataLoader(dset_train, batch_size=setup["batch_size"],
                          shuffle=True, num_workers=6, collate_fn=collate_fn, drop_last=True)

val_loader = DataLoader(dset_val, batch_size=int(setup["batch_size"]/2),
                        shuffle=False, num_workers=6, collate_fn=collate_fn)

print("classes nb:", len(classes), "number of train models: ", len(
    dset_train), "number of test models: ", len(dset_val), classes)

if setup["mvnetwork"] == 'resnet' and not setup["part_seg_mode"]:
    depth2featdim = {18: 512, 34: 512, 50: 2048, 101: 2048, 152: 2048}
    assert setup["depth"] in list(depth2featdim.keys()), "the requested resnt depth not available"
    mvnetwork = torchvision.models.__dict__[
        "resnet{}".format(setup["depth"])](setup["pretrained"])
    mvnetwork.fc = nn.Sequential()
    mvnetwork = MVAgregate(mvnetwork, agr_type="max",feat_dim=depth2featdim[setup["depth"]], num_classes=len(classes)).cuda()
    print('Using ' + setup["mvnetwork"] + str(setup["depth"]))
elif "vit" in setup["mvnetwork"] and not setup["part_seg_mode"]:
    # if setup["pretrained"]:
    vit = timm.create_model(model_name_from_setup(
        setup), pretrained=setup["pretrained"], embed_layer=MyPatchEmbed)
    vit.head = nn.Sequential()
    vit.pre_logits = torch.nn.Identity()
    # else:
    #     vit = ViT(
    #         image_size=setup["image_size"],
    #         patch_size=setup["patch_size"],
    #         num_classes=len(classes),
    #         dim=setup["mvit_dim"],
    #         depth=setup["depth"],
    #         heads=setup["mvit_heads"],
    #         mlp_dim=setup["mlp_dim"],
    #         dropout=setup["mvit_dropout"],
    #         emb_dropout=setup["emb_dropout"],
    #         pool=setup["vit_agr_type"]
    #     )
    #     # remove the mlp head in order to agregate the views before the MLP of the transformer
    #     vit.mlp_head = nn.Sequential()
    if setup["mvnetwork"] == "vit":
        mvnetwork = MVAgregate(vit, agr_type=setup["mv_agr_type"], feat_dim=setup["mvit_dim"], num_classes=len(classes)).cuda()
        
        # inp = torch.rand((1, setup["nb_views"], 3, setup["image_size"], setup["image_size"])).cuda()
        # avg_time = profile_op(1000, mvnetwork, inp)
        # macs, params = get_model_complexity_info(mvnetwork, (setup["nb_views"], 3, setup["image_size"], setup["image_size"]), as_strings=True, print_per_layer_stat=False, verbose=False)
        # print("####### COST ####### \n","\t", macs,"\t", params,"\t", "{}".format(avg_time*1e3))

    elif setup["mvnetwork"] == "mvit":
        mvnetwork = FullCrossViewAttention(vit, patch_size=setup["patch_size"], num_views=setup["nb_views"], feat_dim=setup["mvit_dim"], num_classes=len(classes)).cuda()
    elif setup["mvnetwork"] == "wvit":
        mvnetwork = WindowCrossViewAttention(vit, patch_size=setup["patch_size"], num_views=setup["nb_views"], num_windows=setup["nb_windows"], feat_dim=setup["mvit_dim"], num_classes=len(
            classes), agr_type=setup["mv_agr_type"]).cuda()
    print('Using ' + setup["mvnetwork"] + str(setup["depth"]))
else:
    mvnetwork = torchvision.models.segmentation.deeplabv3_resnet101(pretrained=setup["pretrained"], num_classes=21)
    mvnetwork = MVPartSegmentation(mvnetwork,num_classes=len(classes), parts_per_class=parts_per_class,parallel_head=setup["parallel_head"])
    # mvnetwork = torchvision.models.segmentation.fcn_resnet50(pretrained=setup["pretrained"], num_classes=len(classes)+1)



mvnetwork.cuda()
cudnn.benchmark = True

print('Running on ' + str(torch.cuda.current_device()))


# Loss and Optimizer
lr = setup["lr"]
n_epochs = setup["epochs"]
view_selector = ViewSelector(setup["nb_views"], selection_type=setup["selection_type"],
                             canonical_elevation=setup["canonical_elevation"],canonical_distance= setup["canonical_distance"],
                             shape_features_size=setup["features_size"], transform_distance=setup["transform_distance"], input_view_noise=False, light_direction=setup["light_direction"]).cuda()
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
    # torch.autograd.set_detect_anomaly(True)

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


        loss.backward()


        models_bag["optimizer"].step()
        if setup["log_metrics"]:
            # step = get_current_step(models_bag["optimizer"])
            writer.add_scalar('Zoom/loss', loss.item(), i +setup["c_epoch"]*train_size)
            # writer.add_scalar('Zoom/MVCNN_vals', list(models_bag["mvnetwork"].parameters())[0].data[0, 0, 0].item(), step)
            writer.add_scalar('Zoom/MVCNN_grads', np.sum(np.array([np.sum(x.grad.cpu().numpy(
            ) ** 2) for x in models_bag["mvnetwork"].parameters()])), i + setup["c_epoch"]*train_size)



        if (i + 1) % setup["print_freq"] == 0:
            print("\tIter [%d/%d] Loss: %.4f" % (i + 1, train_size, loss.item()))
        correct += (predicted.cpu() == targets.cpu()).sum()
        total_loss += loss
        n += 1
    avg_loss = total_loss / n
    avg_train_acc = 100 * correct / total

    return avg_train_acc,avg_loss

def train_part_seg(data_loader, models_bag, setup):
    train_size = len(data_loader)
    correct = 0.0

    total_loss = 0.0
    total = 0
    n = 0
    t = enumerate(data_loader)
    for i, (point_set, cls, seg, parts_range, parts_nb, _) in t:
        models_bag["optimizer"].zero_grad()
        bs = point_set.shape[0]
        colors = []
        if setup["use_normals"]:
            normals = point_set[:, :, 3:6]
            colors = (normals + 1.0) / 2.0
            colors = colors/torch.norm(colors, dim=-1,p=float(setup["color_normal_p"]))[..., None]
            point_set = point_set[:,:,0:3]
        rendered_images, indxs, distance_weight_maps , _, _, _ = auto_render_parts(
            cls, None, point_set, models_bag, setup,color=colors, device=None)
        cls = cls.cuda()
        cls = Variable(cls)
        seg = seg.cuda()
        seg = Variable(seg)
        # if i > 1 :
        #     continue
        # print(torch.unique(seg, False), cls, parts_range, parts_nb,"seg.shape",seg.shape)
        seg = seg + 1 - parts_range[..., None].cuda().to(torch.int)
        # parts_range += 1  # the label 0 is reserved for bacgdround


        labels_2d, pix_to_face_mask = compute_image_segment_label_points(
            point_set, batch_points_labels=seg, rendered_pix_to_point=indxs, rendered_images=rendered_images, setup=setup, device=None)
        labels_2d = Variable(labels_2d)

        rendered_images = Variable(rendered_images)



        outputs = models_bag["mvnetwork"](rendered_images)


        criterion = nn.CrossEntropyLoss(ignore_index=0, reduction="none" if setup["add_loss_factor"] else "mean")
        if setup["parallel_head"]:
            target_mask = torch.arange(0, len(classes))[None, ...].repeat(bs, 1).cuda() == cls
            target = labels_2d.to(torch.long).unsqueeze(2).repeat(1, 1, len(classes), 1, 1) * target_mask[..., None][..., None].unsqueeze(1).to(torch.long)
            loss = criterion(outputs, rearrange(target, 'b V cls h w -> (b V) h w cls'))
            predict_mask = target_mask.unsqueeze(1).unsqueeze(1).unsqueeze(1).unsqueeze(1).to(torch.float).repeat(1, setup["nb_views"],1, 1, 1, 1)
            _, predicted = torch.max(torch.max(outputs.data * rearrange(predict_mask, 'b V C h w cls -> (b V) C h w cls'), dim=-1)[0],dim=1)
        else:
            loss = criterion(outputs, mvtosv(labels_2d.to(torch.long)))
            _, predicted = torch.max(outputs.data, dim=1)
        if setup["add_loss_factor"]:
            cls_freq = [('Airplane', 341), ('Bag', 14), ('Cap', 11), ('Car', 158), ('Chair', 704), ('Earphone', 14), ('Guitar', 159), ('Knife', 80),
                        ('Lamp', 286), ('Laptop', 83), ('Motorbike', 51), ('Mug', 38), ('Pistol', 44), ('Rocket', 12), ('Skateboard', 31), ('Table', 848)]
            loss_factor = batch_classes2weights(cls.squeeze(), cls_freq)
            loss_factor = torch.repeat_interleave(loss_factor,(setup["nb_views"]))[...,None][...,None][...,None]
            loss = 300 * (loss * loss_factor).mean()

        loss.backward()
        total_loss += loss.detach().item()
        n += 1
        total += pix_to_face_mask.sum().item()

        correct += ((predicted == mvtosv(labels_2d.to(torch.long))) & mvtosv(pix_to_face_mask[:,:,0,...])).sum().item()

        models_bag["optimizer"].step()
        if setup["log_metrics"]:
            # step = get_current_step(models_bag["optimizer"])
            writer.add_scalar('Zoom/loss', loss.item(), i +
                            setup["c_epoch"]*train_size)
            # writer.add_scalar('Zoom/MVCNN_vals', list(models_bag["mvnetwork"].parameters())[0].data[0, 0, 0].item(), step)
            # writer.add_scalar('Zoom/MVCNN_grads', np.sum(np.array([np.sum(x.grad.cpu().numpy() ** 2) for x in models_bag["mvnetwork"].parameters()])), i + setup["c_epoch"]*train_size)

        if (i + 1) % setup["print_freq"] == 0:
            print("\tIter [%d/%d] Loss: %.4f" %
                (i + 1, train_size, loss.item()))

    avg_loss = total_loss / n
    avg_train_acc = 100 * correct / total

    return avg_train_acc, avg_loss


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


def evluate_part_seg(data_loader, models_bag,  setup, is_test=False):
    if is_test:
        load_checkpoint(setup, models_bag, setup["weights_file"])

    # Eval
    total = 0.0
    correct = 0.0

    total_loss = 0.0
    n = 0
    # shape_iou_tot = 0.
    # shape_iou_cnt = 0.
    total_empty = 0
    # part_intersect = torch.zeros(len(classes),)
    # part_union = torch.zeros(len(classes), )
    # categ_iou = torch.zeros(len(classes),)
    # categ_union = torch.zeros(len(classes), )
    categ_iou = [[] for _ in range(len(classes))]
    categ_union = [[] for _ in range(len(classes))]
    test_path = '/home/hamdiaj/notebooks/learning_torch/data/MVT/mvit_results/00/ZZZ15/renderings'
    test_indx = 0
    visualize = False
    record = ListDict()

    t = tqdm(enumerate(data_loader), total=len(data_loader))
    for i, (point_set, cls, seg, parts_range, parts_nb, real_points_mask) in t:
        with torch.no_grad():
            # if i > 1 :
            #     continue
            bs = point_set.shape[0]
            colors = []
            if setup["use_normals"]:
                normals = point_set[:, :, 3:6]
                colors = (normals + 1.0) / 2.0
                colors = colors/torch.norm(colors, dim=-1,p=float(setup["color_normal_p"]))[..., None]
                point_set = point_set[:,:,0:3]
            rendered_images, indxs, distance_weight_maps, azim, elev, _ = auto_render_parts(
                cls, None, point_set, models_bag, setup, color=colors, device=None)
            cls = cls.cuda()
            cls = Variable(cls)
            seg = seg.cuda()
            seg = Variable(seg)
            real_points_mask = real_points_mask.cuda()

            # print(torch.unique(seg, False), cls, parts_range, parts_nb,"seg.shape",seg.shape)
            seg = seg + 1 - parts_range[..., None].cuda().to(torch.int)
            parts_range += 1  # the label 0 is reserved for bacgdround

            # save_batch_rendered_images(distance_weight_maps[:,:,0:3,...], test_path, "distance_weight_maps.jpg",)
            labels_2d, pix_to_face_mask  = compute_image_segment_label_points(point_set, batch_points_labels=seg, rendered_pix_to_point=indxs, rendered_images=rendered_images, setup=setup, device=None)
            # print("label2d",labels_2d.shape, torch.unique(labels_2d,True),"seg.shape",seg.shape,"indxs.shape",indxs.shape)


            outputs = models_bag["mvnetwork"](rendered_images) 


            criterion = nn.CrossEntropyLoss(ignore_index=0)
            if setup["parallel_head"]:
                target_mask = torch.arange(0, len(classes))[None, ...].repeat(bs, 1).cuda() == cls
                target = labels_2d.to(torch.long).unsqueeze(2).repeat(1, 1, len(classes), 1, 1) * target_mask[..., None][..., None].unsqueeze(1).to(torch.long)
                loss = criterion(outputs, rearrange(target, 'b V cls h w -> (b V) h w cls'))
                predict_mask = target_mask.unsqueeze(1).unsqueeze(1).unsqueeze(1).unsqueeze(1).to(torch.float).repeat(1, setup["nb_views"], 1, 1, 1, 1)
                feats = torch.max(outputs.data * rearrange(predict_mask, 'b V C h w cls -> (b V) C h w cls'), dim=-1)[0]
            else:
                feats = outputs.data
            _, predicted = torch.max(feats, dim=1)

            if "attention" in setup["lifting_method"]:
                dir_vec = torch_direction_vector(batch_tensor(azim, dim=1, squeeze=True), batch_tensor(elev, dim=1, squeeze=True))
                dir_vec = unbatch_tensor(dir_vec, dim=1, unsqueeze=True, batch_size=bs)
                if setup["lifting_method"] == "point_attention":
                    points_weights = torch.abs(torch.bmm(dir_vec.cuda(),normals.transpose(1,2).cuda()))
                    views_weights = points_weights.sum(dim=-1)
                    views_weights = views_weights / views_weights.sum(dim=-1,keepdim=True)
                    views_weights = views_weights[..., None][..., None][..., None]
                elif setup["lifting_method"] == "pixel_attention":
                    normal_weight_maps = dir_vec.cuda()[..., None][..., None].repeat(1,1,1,setup["image_size"],setup["image_size"]) * rendered_images
                    views_weights = torch.abs(normal_weight_maps.sum(dim=2, keepdim=True))
                elif setup["lifting_method"] == "view_attention":
                    normal_weight_maps = dir_vec.cuda()[..., None][..., None].repeat(1,1,1,setup["image_size"],setup["image_size"]) * rendered_images
                    views_weights = torch.abs(normal_weight_maps.sum(dim=2, keepdim=True)).sum(dim=3, keepdim=True).sum(dim=4, keepdim=True)
                    views_weights = views_weights / views_weights.sum(dim=1,keepdim=True)

            else :
                views_weights = torch.ones_like(azim).cuda()[..., None][..., None][..., None]
            predictions_3d = lift_2D_to_3D(point_set, predictions_2d=svctomvc(feats, nb_views=setup["nb_views"]), rendered_pix_to_point=indxs, views_weights=views_weights, parts_range=parts_range, parts_nb=parts_nb, lifting_method=setup["lifting_method"])
            predictions_3d = post_process_segmentation(point_set, predictions_3d, iterations=setup["post_process_iters"],K_neighbors=setup["post_process_k"])
            # predictions_3d = lift_2D_to_3D(point_set, predictions_2d=labels_2d, rendered_pix_to_point=indxs,views_weights=views_weights, parts_range=parts_range, parts_nb=parts_nb,lifting_method=setup["lifting_method"],  setup=setup, device=None)
            # if visualize :
            #     gt_images_path = os.path.join(test_path, "GT_renderings_{}.png".format(str(cls[test_indx].item())))
            #     pred_images_path = os.path.join(test_path, "GT_renderings_{}_pred.png".format(str(cls[test_indx].item())))
            #     _ = view_ptc_labels(rotation_matrix([1, 0, 0], 90).dot(point_set[test_indx].cpu().numpy().T).T, seg[test_indx].cpu().numpy(),COLOR_LABEL_VALUES_LIST, size=0.01, save_name=gt_images_path)
            #     _ = view_ptc_labels(rotation_matrix([1, 0, 0], 90).dot(point_set[test_indx].cpu().numpy(
            #     ).T).T, predictions_3d[test_indx].cpu().numpy(), COLOR_LABEL_VALUES_LIST, size=0.01, save_name=pred_images_path)
            #     save_batch_rendered_segmentation_images(labels_2d, test_path, "2d_labels.jpg")
            #     save_batch_rendered_segmentation_images(
            #         svtomv(predicted, nb_views=setup["nb_views"]), test_path, "2d_predictions.jpg")
            #     save_batch_rendered_images(rendered_images[:, :, 0:3, ...], test_path, "original.jpg",)
            
            total_loss += loss.detach().item()
            n += 1
            total += real_points_mask.sum().item()  # seg.size(0)*seg.size(1)
            total_empty += ((predictions_3d == 0) &real_points_mask.to(torch.bool)).sum().item()

            correct += ((predictions_3d == seg) & real_points_mask.to(torch.bool)).sum().item()
 
            # IOU calculations
            cur_shape_miou = batch_points_mIOU(seg - 1, predictions_3d - 1, real_points_mask.to(torch.bool), parts=parts_nb,)
            # print(cur_shape_miou.shape)
            if setup["record_extra_metrics"]:
                pixel_perc, point_perc, iou, valid_iou, cls_nb, part_nb = extra_IOU_metrics(
                    seg - 1, predictions_3d - 1, labels_2d-1, pix_to_face_mask, real_points_mask.to(torch.bool), cls, parts=parts_nb,)
                c_record = ListDict({"valid_iou":valid_iou,"cls_nb":cls_nb, "part_nb":part_nb,"pixel_perc": pixel_perc, "point_perc": point_perc, "iou": iou})
                record.extend(c_record)
                save_results(setup["views_file"], record)
            for cat in range(len(classes)):
                cat_cur_shape_miou = cur_shape_miou[ (cls == cat).view(-1)]
                categ_iou[cat] += cat_cur_shape_miou.cpu().numpy().tolist()


                # categ_iou[cat] += (I * (cls == cat ).to(torch.long)  ).sum()
                # categ_union[cat] += (U * (cls == cat).to(torch.long)).sum()
            # shape_iou_tot += cur_shape_miou.sum().item()
            # shape_iou_cnt += bs
    # shape_mIoU = 100 * shape_iou_tot / shape_iou_cnt
    # part_iou = part_intersect/ part_union
    print("The number of objects per class: ", list(zip(classes, [len(x) for x in categ_iou])))
    all_ious = []
    mean_cat_iou = []
    for cat in range(len(classes)):
        all_ious += categ_iou[cat]
        mean_cat_iou.append(np.mean(np.array(categ_iou[cat]), axis=-1))
    mean_inst_iou = 100 * np.mean(np.array(all_ious))
    # cat_iou = categ_iou / categ_union
    print("The mIOU per class average: ", list(zip(classes, mean_cat_iou)))

    mean_cat_iou = 100 * np.mean(np.array(mean_cat_iou))
    # mean_inst_iou = 100.*  part_iou

        


    avg_loss = total_loss / n
    avg_test_acc = 100 * correct / float(total)
    point_coverage = 100 - 100 * total_empty / float(total) 

    return avg_test_acc, mean_cat_iou, mean_inst_iou, avg_loss, point_coverage

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
if setup["resume"] or setup["test_only"] or setup["test_retrieval_only"]:
    load_checkpoint(setup, models_bag, setup["weights_file"])

if setup["train_only"] :
    if setup["log_metrics"]:
        writer = SummaryWriter(setup["logs_dir"])
        writer.add_hparams(setup, {"hparams/best_acc": 0.0})

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



        # Log epoch to tensorboard
        # See log using: tensorboard --logdir='logs' --port=6006 ######################################
        # util.logEpoch(logger, mvnetwork, epoch + 1, avg_loss, avg_test_acc) #############################################
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
        if avg_test_acc.item() >= setup["best_acc"] :
            print('\tSaving checkpoint - Acc: %.2f' % avg_test_acc)
            saveables["best_acc"] = avg_test_acc
            setup["best_loss"] = avg_loss.item()
            setup["best_acc"] = avg_test_acc.item()
            save_checkpoint(saveables, setup, views_record,
                            setup["weights_file"], ignore_saving_models=setup["ignore_saving_models"])

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

    # avg_train_acc, avg_train_loss = evluate(train_loader, models_bag, setup)
    # print('\ttrain Acc: %.2f - val Loss: %.4f' %(avg_train_acc.item(), avg_train_loss.item()))

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
        # extra_info = torch.from_numpy(extra_info)
        auto_render_and_save_images_and_cameras(targets, meshes, extra_info,correction_factor, images_path=images_path,
                                                cameras_path=cameras_path, models_bag=models_bag, setup=setup, device=None)
if setup["test_retrieval_only"]:
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
    

elif setup["part_seg_mode"]:
    is_test = False
    if setup["log_metrics"]:
        writer = SummaryWriter(setup["logs_dir"])
        writer.add_hparams(setup, {"hparams/best_acc": 0.0 , "hparams/best_inst_iou": 0.0,"hparams/best_cat_iou": 0.0})
    setup["best_inst_iou"] = 0.0
    setup["best_cat_iou"] = 0.0

    for epoch in range(setup["start_epoch"], n_epochs):
        setup["c_epoch"] = epoch
        print('\n-----------------------------------')
        print('Epoch: [%d/%d]' % (epoch+1, n_epochs))
        start = time.time()
        if not setup["test_part_seg_only"]:
            models_bag["mvnetwork"].train()
            models_bag["view_selector"].train()
            models_bag["feature_extractor"].train()
            avg_train_acc, avg_train_loss = train_part_seg(train_loader, models_bag, setup)
            print('Time taken: %.2f sec.' % (time.time() - start))
            print('\ttrain pixel acc: %.2f - train 2D Loss: %.4f ' %(avg_train_acc, avg_train_loss))
            if setup["log_metrics"]:
                writer.add_scalar('Loss/train', avg_train_loss, epoch)
                writer.add_scalar('Accuracy/train', avg_train_acc, epoch)
        else:
            is_test = True
        models_bag["mvnetwork"].eval()
        models_bag["view_selector"].eval()
        models_bag["feature_extractor"].eval()
        avg_test_acc, mean_cat_iou_test, mean_inst_iou_test, avg_loss, point_coverage = evluate_part_seg(
            val_loader, models_bag, setup, is_test=is_test)
        print('\nEvaluation:')
        print('\tVal point Acc: %.2f - val category-avg iou: %.2f - val instance-avg iou: %.2f - val Coverage: %.2f - val 2D Loss: %.3f ' %
              (avg_test_acc, mean_cat_iou_test, mean_inst_iou_test, point_coverage,avg_loss))
        # print(mean_inst_iou_test)
        print('\tCurrent best acc: {:.2f}  - best category_avg miou: {:.2f} - best instance_avg miou: {:.2f}'.format(
            setup["best_acc"], setup["best_cat_iou"], setup["best_inst_iou"]))
        if setup["log_metrics"]:
            writer.add_scalar('Loss/val', avg_loss, epoch)
            writer.add_scalar('Accuracy/val', avg_test_acc, epoch)
            writer.add_scalar('test_mIOU/categ', mean_cat_iou_test, epoch)
            writer.add_scalar('test_mIOU/inst',mean_inst_iou_test, epoch)


        saveables = {'epoch': epoch + 1,
                     'state_dict': models_bag["mvnetwork"].state_dict(),
                     "view_selector": models_bag["view_selector"].state_dict(),
                     "feature_extractor": models_bag["feature_extractor"].state_dict(),
                     'acc': avg_test_acc,
                     'best_acc': setup["best_acc"],
                     'best_inst_iou': setup["best_inst_iou"],
                     'best_cat_iou': setup["best_cat_iou"],
                     'optimizer': models_bag["optimizer"].state_dict(),
                     }
        if setup["save_all"]:
            save_checkpoint(saveables, setup, None,
                            setup["weights_file"])
        # Save mvnetwork
        if mean_cat_iou_test > setup["best_cat_iou"]:
            print('\tSaving checkpoint - Acc: %.2f' % avg_test_acc)
            saveables["best_acc"] = avg_test_acc
            setup["best_loss"] = avg_loss
            setup["best_acc"] = avg_test_acc
            setup["point_coverage"] = point_coverage
            setup["best_cat_iou"] = mean_cat_iou_test
            setup["best_inst_iou"] = mean_inst_iou_test
            save_checkpoint(saveables, setup, None,setup["weights_file"], ignore_saving_models=setup["ignore_saving_models"])

        # Decaying Learning Rate
        if (epoch + 1) % setup["lr_decay_freq"] == 0:
            lr *= setup["lr_decay"]
            models_bag["optimizer"] = torch.optim.AdamW(
                models_bag["mvnetwork"].parameters(), lr=lr)
            print('Learning rate:', lr)
        if (epoch + 1) % setup["plot_freq"] == 0 or setup["test_part_seg_only"]:
            for indx, ii in enumerate(PLOT_SAMPLE_NBS):
                bs = 1 
                (point_set, cls, seg, parts_range,parts_nb, real_points_mask) = dset_val[ii]
                given_labels = list(range(0, parts_nb),)
                point_set = torch.from_numpy(point_set)[None, ...]
                colors = []
                if setup["use_normals"]:
                    normals = point_set[:,:,3:6]
                    colors = (normals + 1.0) / 2.0
                    colors = colors/torch.norm(colors, dim=-1,p=float(setup["color_normal_p"]))[..., None]
                    point_set = point_set[:,:,0:3]
                seg = torch.from_numpy(seg)[None, ...].cuda()
                real_points_mask = torch.from_numpy(real_points_mask)[
                    None, ...].cuda()
                parts_range = torch.Tensor([parts_range]).to(torch.int)
                parts_nb = torch.Tensor([parts_nb]).to(torch.int)
                cls = torch.Tensor([cls]).cuda()
                seg = seg + 1 - parts_range[..., None].cuda().to(torch.int)
                parts_range += 1  # the label 0 is reserved for bacgdround


                renderings_root_folder = os.path.join(setup["renderings_dir"], str(indx))
                check_folder(renderings_root_folder)

                rendered_images, indxs, distance_weight_maps, azim, elev, _ = auto_render_parts(
                    cls, None, point_set, models_bag, setup,color=colors, device=None)
                labels_2d, pix_to_face_mask  = compute_image_segment_label_points(point_set, batch_points_labels=seg, rendered_pix_to_point=indxs, rendered_images=rendered_images, setup=setup, device=None)


                outputs = models_bag["mvnetwork"](rendered_images)
                if setup["parallel_head"]:
                    target_mask = torch.arange(0, len(classes))[
                        None, ...].repeat(bs, 1).cuda() == cls
                    predict_mask = target_mask.unsqueeze(1).unsqueeze(1).unsqueeze(
                        1).unsqueeze(1).to(torch.float).repeat(1, setup["nb_views"], 1, 1, 1, 1)
                    feats = torch.max(outputs.data * rearrange(predict_mask, 'b V C h w cls -> (b V) C h w cls'), dim=-1)[0]
                else:
                    feats = outputs.data
                _, predicted = torch.max(feats, dim=1)

                if setup["lifting_method"] == "attention":
                    dir_vec = torch_direction_vector(batch_tensor(azim, dim=1, squeeze=True), batch_tensor(elev, dim=1, squeeze=True))
                    dir_vec = unbatch_tensor(dir_vec, dim=1, unsqueeze=True, batch_size=bs)
                    points_weights = torch.abs(torch.bmm(dir_vec.cuda(),normals.transpose(1,2).cuda()))
                    views_weights = points_weights.sum(dim=-1)
                    views_weights = views_weights/views_weights.sum(dim=-1)[...,None] 
                else :
                    views_weights = torch.ones_like(azim).cuda()

                predictions_3d = lift_2D_to_3D(point_set, predictions_2d=svctomvc(feats, nb_views=setup["nb_views"]), rendered_pix_to_point=indxs,views_weights=views_weights, parts_range=parts_range, parts_nb=parts_nb,lifting_method=setup["lifting_method"])
                predictions_3d = post_process_segmentation(
                    point_set, predictions_3d, iterations=setup["post_process_iters"], K_neighbors=setup["post_process_k"])
                predictions_3d_projected ,_= compute_image_segment_label_points(point_set, batch_points_labels=predictions_3d.to(torch.int), rendered_pix_to_point=indxs, rendered_images=rendered_images, setup=setup, device=None)
                save_batch_rendered_segmentation_images(labels_2d, renderings_root_folder, "GT_renderings_{}.jpg".format(str(epoch + 1)),)
                save_batch_rendered_segmentation_images(svtomv(
                    predicted, nb_views=setup["nb_views"])* (~pix_to_face_mask).to(torch.long)[:,:,0,...],
                    renderings_root_folder, "PRED_2D_renderings_{}.jpg".format(str(epoch + 1)), given_labels=given_labels)
                save_batch_rendered_segmentation_images(predictions_3d_projected, renderings_root_folder, "PRED_3D_renderings_{}.jpg".format(str(epoch + 1)))
                save_batch_rendered_images(rendered_images[:, :, 0:3, ...], renderings_root_folder, "original_renderings_{}.jpg".format(str(epoch + 1)),)
                if setup["test_part_seg_only"]:
                    _ = view_ptc_labels(rotation_matrix([1, 0, 0], 90).dot(point_set[0].cpu().numpy().T).T, seg[0].cpu().numpy(),COLOR_LABEL_VALUES_LIST, size=0.01)
                    _ = view_ptc_labels(rotation_matrix([1, 0, 0], 90).dot(point_set[0].cpu().numpy().T).T, predictions_3d[0].cpu().numpy(), COLOR_LABEL_VALUES_LIST, size=0.01)

                cur_shape_miou = batch_points_mIOU(seg - 1, predictions_3d - 1, real_points_mask.to(torch.bool), parts=parts_nb,)
                print("object {} of class {} has mIOU: {:.1f}".format(ii,classes[int(cls.item())],100*cur_shape_miou.item()))

            if setup["test_part_seg_only"]:
                raise Exception("finshed testing ")

    
    if setup["log_metrics"]:
        writer.add_hparams(setup, {
                           "hparams/best_acc": setup["best_acc"], "hparams/best_inst_iou": setup["best_inst_iou"], "hparams/best_cat_iou": setup["best_cat_iou"]})

elif setup["scene_seg_mode"]:
    is_test = False
    if setup["log_metrics"]:
        writer = SummaryWriter(setup["logs_dir"])
        writer.add_hparams(setup, {"hparams/best_acc": 0.0 , "hparams/best_iou": 0.0})
    setup["best_iou"] = 0.0
    test_path = '/home/hamdiaj/Downloads/birmingham_block_0.ply'
    COLOR_LABEL_VALUES_LIST = [(141,211,199),(255,255,179),(190,186,218),(251,128,114),(128,177,211),(253,180,98),(179,222,105),(252,205,229),(217,217,217),(188,128,189),(204,235,197),(255,237,111),(0,0,0),(255,0,0),(0,255,0),(0,0,255)]
    COLOR_LABEL_VALUES_LIST = np.array(COLOR_LABEL_VALUES_LIST)/255.0 
    COLOR_LABEL_VALUES_LIST = COLOR_LABEL_VALUES_LIST.tolist()
    pp = trimesh.load(test_path)
    point_set = np.array(pp.vertices)
    point_set = rotation_matrix([1, 0, 0], -90).dot(point_set.T).T
    colors = np.array(pp.colors)
    nb_points = pp.metadata["ply_raw"]["vertex"]["length"]
    print("scene has {} points ".format(nb_points))
    labels = np.asarray(tuple(pp.metadata["ply_raw"]["vertex"]["data"].tolist()))[:, 6]
    # viewer = view_ptc_labels(point_set, labels, COLOR_LABEL_VALUES_LIST, size=0.02)
    # viewer.attributes(colors/255.0)
    # viewer.color_map('cool')
    point_set = torch.from_numpy(point_set)
    point_set = torch_center_and_normalize(point_set, p=2)[None, ...]
    cls = torch.Tensor([1])[None, ...]
    colors = torch.from_numpy(colors)[None, :,0:3].to(torch.float)
    seg = torch.from_numpy(labels)[None, ...].cuda().to(torch.int32)
    given_labels = list(range(0, len(torch.unique(seg))),)

    # real_points_mask = torch.from_numpy(real_points_mask)[
    #     None, ...].cuda()
    # parts_nb = torch.Tensor([parts_nb]).to(torch.int)
    seg = seg + 1 
    epoch = 0


    renderings_root_folder = os.path.join(setup["renderings_dir"], str(100))
    check_folder(renderings_root_folder)

    rendered_images, indxs, distance_weight_maps, _,_,_ = auto_render_parts(
        cls, None, point_set, models_bag, setup, color=colors, device=None)
    save_batch_rendered_images(rendered_images[:, :, 0:3, ...].to(torch.int), renderings_root_folder, "original_renderings_{}.jpg".format(str(epoch + 1)),)

    labels_2d, pix_to_face_mask  = compute_image_segment_label_points(point_set, batch_points_labels=seg, rendered_pix_to_point=indxs, rendered_images=rendered_images, setup=setup, device=None)
    save_batch_rendered_segmentation_images(labels_2d, renderings_root_folder, "GT_renderings_{}.jpg".format(str(epoch + 1)),)





