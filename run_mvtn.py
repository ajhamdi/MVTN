import torch
import torch.nn as nn
import sys
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.autograd import Variable


import scipy
import scipy.spatial


from tqdm import tqdm
import pickle as pkl


import torchvision.transforms as transforms
import torchvision

import argparse
import numpy as np
import time
import os


from util import *
from ops import *

from models.pointnet import *
from models.mvtn import *
from models.multi_view import *
from models.renderer import *


from torch.utils.tensorboard import SummaryWriter
from custom_dataset import ModelNet40, collate_fn, ShapeNetCore, ScanObjectNN
from rotationNet.mvt_rotnet import RotationNet, AverageMeter, my_accuracy
from viewGCN.tools.Trainer_mvt import ModelNetTrainer_mvt

from viewGCN.model.view_gcn import view_GCN, SVCNN

PLOT_SAMPLE_NBS = [242, 7, 549, 112, 34]


parser = argparse.ArgumentParser(description='MVTN-PyTorch')

parser.add_argument('--data_dir', required=True,  help='path to 3D dataset')
parser.add_argument('--run_mode', '-rmode',  default="train", choices=["train", "test_cls", "test_retr", "test_rot", "test_occ"],
                    help='The mode of running the code: train, test classification, test retrieval, test rotation robustness, or test occlusion robustness. You have to train before testing')
parser.add_argument('--mvnetwork', '-m',  default="mvcnn", choices=["mvcnn", "rotnet", "viewgcn"],
                    help='the type of multi-view network used:')
parser.add_argument('--nb_views', type=int,
                    help='number of views in the multi-view setup')
parser.add_argument('--views_config', '-s',  default="circular", choices=["circular", "random", "learned_circular", "learned_direct", "spherical", "learned_spherical", "learned_random", "learned_transfer", "custom"],
                    help='the selection type of views ')
parser.add_argument('--gpu', type=int,
                    default=0, help='GPU number ')
parser.add_argument('--dset_variant', '-dsetp', help='The variant used of the `ScanObjectNN` dataset  ',
                    default="obj_only", choices=["obj_only", "with_bg", "hardest"])
parser.add_argument('--pc_rendering', dest='pc_rendering',
                    action='store_true', help='use point cloud renderer instead of mesh renderer  ')
parser.add_argument('--object_color', '-clr',  default="white", choices=["white", "random", "black", "red", "green", "blue", "custom"],
                    help='the selection type of views ')
parser.add_argument('--epochs', default=100, type=int,
                    help='number of total epochs to run (default: 100)')
parser.add_argument('--batch_size', '-b', default=20, type=int,
                    help='mini-batch size (default: 20)')
parser.add_argument('-r', '--resume', dest='resume',
                    action='store_true', help='continue training from the `setup[weights_file] checkpoint ')
parser.add_argument("--viewgcn_phase", default="all", choices=["all", "first", "second"],
                    help='what stage of training of the ViewGCN ( it has two stages)')
parser.add_argument('--config_file', '-cfg',  default="config.yaml", help='the conifg yaml file for more options.')


args = parser.parse_args()
args = vars(args)
config = read_yaml(args["config_file"])
setup = {**args, **config}
if setup["mvnetwork"] in ["rotnet", "mvcnn"]:
    initialize_setup(setup)
else:
    initialize_setup_gcn(setup)

print('Loading data')


torch.cuda.set_device(int(setup["gpu"]))
if "modelnet" in setup["data_dir"].lower():
    dset_train = ModelNet40(setup["data_dir"], "train", nb_points=setup["nb_points"], simplified_mesh=setup["simplified_mesh"], cleaned_mesh=setup["cleaned_mesh"], dset_norm=setup["dset_norm"], return_points_saved=setup["return_points_saved"],
                            is_rotated=setup["rotated_train"])
    dset_val = ModelNet40(setup["data_dir"], "test", nb_points=setup["nb_points"], simplified_mesh=setup["simplified_mesh"], cleaned_mesh=setup["cleaned_mesh"], dset_norm=setup["dset_norm"], return_points_saved=setup["return_points_saved"],
                          is_rotated=setup["rotated_test"])
    classes = dset_train.classes

elif "shapenetcore" in setup["data_dir"].lower():
    dset_train = ShapeNetCore(setup["data_dir"], ("train",), setup["nb_points"], load_textures=False,
                              dset_norm=setup["dset_norm"], simplified_mesh=setup["simplified_mesh"])
    dset_val = ShapeNetCore(setup["data_dir"], ("test",), setup["nb_points"], load_textures=False,
                            dset_norm=setup["dset_norm"], simplified_mesh=setup["simplified_mesh"])

    classes = dset_val.classes
elif "scanobjectnn" in setup["data_dir"].lower():
    dset_train = ScanObjectNN(setup["data_dir"], 'train',  setup["nb_points"],
                              variant=setup["dset_variant"], dset_norm=setup["dset_norm"])
    dset_val = ScanObjectNN(setup["data_dir"], 'test',  setup["nb_points"],
                            variant=setup["dset_variant"], dset_norm=setup["dset_norm"])
    classes = dset_train.classes

train_loader = DataLoader(dset_train, batch_size=setup["batch_size"],
                          shuffle=True, num_workers=6, collate_fn=collate_fn, drop_last=True)

val_loader = DataLoader(dset_val, batch_size=int(setup["batch_size"]),
                        shuffle=False, num_workers=6, collate_fn=collate_fn)

print("classes nb:", len(classes), "number of train models: ", len(
    dset_train), "number of test models: ", len(dset_val), classes)

if setup["mvnetwork"] == "mvcnn":
    depth2featdim = {18: 512, 34: 512, 50: 2048, 101: 2048, 152: 2048}
    assert setup["depth"] in list(
        depth2featdim.keys()), "the requested resnt depth not available"
    mvnetwork = torchvision.models.__dict__[
        "resnet{}".format(setup["depth"])](setup["pretrained"])
    mvnetwork.fc = nn.Sequential()
    mvnetwork = MVAgregate(mvnetwork, agr_type="max",
                           feat_dim=depth2featdim[setup["depth"]], num_classes=len(classes))
    print('Using ' + setup["mvnetwork"] + str(setup["depth"]))
if setup["mvnetwork"] == "rotnet":
    mvnetwork = torchvision.models.__dict__["resnet{}".format(
        setup["depth"])](pretrained=setup["pretrained"])
    mvnetwork = RotationNet(mvnetwork, "resnet{}".format(
        setup["depth"]), (len(classes)+1) * setup["nb_views"])
if setup["mvnetwork"] == "viewgcn":
    mvnetwork = SVCNN(setup["exp_id"], nclasses=len(
        classes), pretraining=setup["pretrained"], cnn_name=setup["cnn_name"])

mvnetwork.cuda()
cudnn.benchmark = True

print('Running on ' + str(torch.cuda.current_device()))


lr = setup["learning_rate"]
n_epochs = setup["epochs"]


mvtn = MVTN(setup["nb_views"], views_config=setup["views_config"],
            canonical_elevation=setup["canonical_elevation"], canonical_distance=setup["canonical_distance"],
            shape_features_size=setup["features_size"], transform_distance=setup["transform_distance"], input_view_noise=setup["input_view_noise"], shape_extractor=setup["shape_extractor"], screatch_feature_extractor=setup["screatch_feature_extractor"]).cuda()
mvrenderer = MVRenderer(nb_views=setup["nb_views"], image_size=setup["image_size"], pc_rendering=setup["pc_rendering"], object_color=setup["object_color"], background_color=setup["background_color"],
                        faces_per_pixel=setup["faces_per_pixel"], points_radius=setup["points_radius"],  points_per_pixel=setup["points_per_pixel"], light_direction=setup["light_direction"], cull_backfaces=setup["cull_backfaces"])
print(setup)
criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(
    mvnetwork.parameters(), lr=lr, weight_decay=setup["weight_decay"])
if setup["is_learning_views"]:
    mvtn_optimizer = torch.optim.AdamW(mvtn.parameters(
    ), lr=setup["mvtn_learning_rate"], weight_decay=setup["mvtn_weight_decay"])
else:
    mvtn_optimizer = None


models_bag = {"mvnetwork": mvnetwork, "optimizer": optimizer,
              "mvtn": mvtn, "mvtn_optimizer": mvtn_optimizer, "mvrenderer": mvrenderer}


def train(data_loader, models_bag, setup):
    train_size = len(data_loader)
    total = 0.0
    correct = 0.0

    total_loss = 0.0
    n = 0

    for i, (targets, meshes, points) in enumerate(data_loader):
        c_batch_size = targets.shape[0]
        models_bag["optimizer"].zero_grad()
        if setup["is_learning_views"]:
            models_bag["mvtn_optimizer"].zero_grad()

        azim, elev, dist = models_bag["mvtn"](
            points, c_batch_size=c_batch_size)
        rendered_images, _ = models_bag["mvrenderer"](
            meshes, points,  azim=azim, elev=elev, dist=dist)
        rendered_images = regualarize_rendered_views(
            rendered_images, setup["view_reg"], setup["augment_training"], setup["crop_ratio"])
        targets = targets.cuda()
        targets = Variable(targets)
        outputs = models_bag["mvnetwork"](rendered_images)[0]
        loss = criterion(outputs, targets)
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)

        loss.backward()

        models_bag["optimizer"].step()
        if setup["is_learning_views"]:
            models_bag["mvtn_optimizer"].step()
            if setup["clip_grads"]:
                clip_grads_(models_bag["mvtn"].parameters(),
                            setup["mvtn_clip_grads_value"])
            if setup["log_metrics"]:
                step = get_current_step(models_bag["mvtn_optimizer"])
                writer.add_scalar('Zoom/loss', loss.item(), step)
                writer.add_scalar(
                    'Zoom/MVT_vals', list(models_bag["mvtn"].parameters())[0].data[0, 0].item(), step)
                writer.add_scalar('Zoom/MVT_grads', np.sum(np.array(
                    [np.sum(x.grad.cpu().numpy() ** 2) for x in models_bag["mvtn"].parameters()])), step)
                writer.add_scalar(
                    'Zoom/MVCNN_vals', list(models_bag["mvnetwork"].parameters())[0].data[0].item(), step)
                writer.add_scalar('Zoom/MVCNN_grads', np.sum(np.array([np.sum(
                    x.grad.cpu().numpy() ** 2) for x in models_bag["mvnetwork"].parameters()])), step)

        if (i + 1) % setup["print_freq"] == 0:
            print("\tIter [%d/%d] Loss: %.4f" %
                  (i + 1, train_size, loss.item()))
        correct += (predicted.cpu() == targets.cpu()).sum()
        total_loss += loss.item()
        n += 1
    avg_loss = total_loss / n
    avg_train_acc = 100 * correct / total

    return avg_train_acc, avg_loss


def train_rotationNet(data_loader, models_bag, setup):
    train_size = len(data_loader)

    total_loss = 0.0
    n = 0
    top1 = AverageMeter()

    for i, (targets, meshes, points) in enumerate(data_loader):
        models_bag["optimizer"].zero_grad()
        if setup["is_learning_views"]:
            models_bag["mvtn_optimizer"].zero_grad()

        c_batch_size = targets.shape[0]
        azim, elev, dist = models_bag["mvtn"](
            points, c_batch_size=c_batch_size)
        rendered_images, _ = models_bag["mvrenderer"](
            meshes, points,  azim=azim, elev=elev, dist=dist)
        rendered_images = regualarize_rendered_views(
            rendered_images, setup["view_reg"], setup["augment_training"], setup["crop_ratio"])

        targets = targets.repeat_interleave((setup["nb_views"])).cuda()

        input_var = mvctosvc(rendered_images).cuda()
        targets_ = torch.LongTensor(targets.size(0) * setup["nb_views"])

        output = models_bag["mvnetwork"](input_var)
        num_classes = int(output.size(1) / setup["nb_views"]) - 1
        output = output.view(-1, num_classes + 1)

        output_ = torch.nn.functional.log_softmax(output, dim=-1)

        output_ = output_[
            :, :-1] - torch.t(output_[:, -1].repeat(1, output_.size(1)-1).view(output_.size(1)-1, -1))

        output_ = output_.view(-1, setup["nb_views"]
                               * setup["nb_views"], num_classes)
        output_ = output_.data.cpu().numpy()
        output_ = output_.transpose(1, 2, 0)

        for j in range(targets_.size(0)):
            targets_[j] = num_classes

        scores = np.zeros((vcand.shape[0], num_classes, c_batch_size))
        for j in range(vcand.shape[0]):
            for k in range(vcand.shape[1]):
                scores[j] = scores[j] + \
                    output_[vcand[j][k] * setup["nb_views"] + k]

        for n in range(c_batch_size):
            j_max = np.argmax(scores[:, targets[n * setup["nb_views"]], n])

            for k in range(vcand.shape[1]):
                targets_[n * setup["nb_views"] * setup["nb_views"] + vcand[j_max]
                         [k] * setup["nb_views"] + k] = targets[n * setup["nb_views"]]

        targets_ = targets_.cuda()
        targets_var = torch.autograd.Variable(targets_)

        loss = criterion(output, targets_var)

        loss.backward()

        models_bag["optimizer"].step()
        if setup["is_learning_views"]:
            models_bag["mvtn_optimizer"].step()
            if setup["clip_grads"]:
                clip_grads_(models_bag["mvtn"].parameters(),
                            setup["mvtn_clip_grads_value"])
            if setup["log_metrics"]:
                step = get_current_step(models_bag["mvtn_optimizer"])
                writer.add_scalar('Zoom/loss', loss.item(), step)
                writer.add_scalar(
                    'Zoom/MVTN_vals', list(models_bag["mvtn"].parameters())[0].data[0, 0].item(), step)
                writer.add_scalar('Zoom/MVT_grads', np.sum(np.array([np.sum(x.grad.cpu(
                ).numpy() ** 2) for x in models_bag["mvtn"].parameters()])), step)
                writer.add_scalar(
                    'Zoom/MVCNN_vals', list(models_bag["mvnetwork"].parameters())[0].data[0, 0, 0, 0].item(), step)
                writer.add_scalar('Zoom/MVCNN_grads', np.sum(np.array([np.sum(
                    x.grad.cpu().numpy() ** 2) for x in models_bag["mvnetwork"].parameters()])), step)

        output = output[:, :-1] - torch.t(output[:, -1].repeat(
            1, output.size(1)-1).view(output.size(1)-1, -1))
        output = output.view(-1, setup["nb_views"]
                             * setup["nb_views"], num_classes)
        prec1, _ = my_accuracy(output.data, targets, vcand,
                               setup["nb_views"], topk=(1, 5))
        top1.update(prec1.item(), c_batch_size)

        if (i + 1) % setup["print_freq"] == 0:
            print("\tIter [%d/%d] Loss: %.4f" %
                  (i + 1, train_size, loss.item()))

        total_loss += loss.item()
        n += 1
    avg_loss = total_loss / n

    return top1.avg, avg_loss


def evaluate_rotationNet(data_loader, models_bag, setup):
    train_size = len(data_loader)

    total_loss = 0.0
    n = 0
    top1 = AverageMeter()
    t = tqdm(enumerate(data_loader), total=len(data_loader))
    for i, (targets, meshes, points) in t:
        with torch.no_grad():

            c_batch_size = targets.shape[0]
            azim, elev, dist = models_bag["mvtn"](
                points, c_batch_size=c_batch_size)
            rendered_images, _ = models_bag["mvrenderer"](
                meshes, points,  azim=azim, elev=elev, dist=dist)
            targets = targets.repeat_interleave((setup["nb_views"])).cuda()

            input_var = torch.autograd.Variable(
                mvctosvc(rendered_images)).cuda()
            target_var = torch.autograd.Variable(targets)

            output = models_bag["mvnetwork"](input_var)
            loss = criterion(output, target_var)

            num_classes = int(output.size(1) / setup["nb_views"]) - 1
            output = output.view(-1, num_classes + 1)
            output = torch.nn.functional.log_softmax(output, dim=-1)
            output = output[:, :-1] - torch.t(output[:, -1].repeat(
                1, output.size(1)-1).view(output.size(1)-1, -1))
            output = output.view(-1, setup["nb_views"]
                                 * setup["nb_views"], num_classes)
            output = output.view(-1, setup["nb_views"]
                                 * setup["nb_views"], num_classes)
            prec1, _ = my_accuracy(output.data, targets, vcand,
                                   setup["nb_views"], topk=(1, 5))
            top1.update(prec1.item(), c_batch_size)

            total_loss += loss.item()
            n += 1
    avg_loss = total_loss / n

    return top1.avg, avg_loss


def evluate(data_loader, models_bag,  setup, is_test=False, retrieval=False):
    if is_test:
        load_checkpoint(setup, models_bag, setup["weights_file"])

    total = 0.0
    correct = 0.0

    total_loss = 0.0
    n = 0
    if retrieval:
        features_training = np.load(setup["feature_file"])
        targets_training = np.load(setup["targets_file"])
        N_retrieved = 1000 if "shapenetcore" in setup["data_dir"].lower() else len(
            features_training)

        features_training = lfda.transform(features_training)

        kdtree = scipy.spatial.KDTree(features_training)
        all_APs = []

    views_record = ListDict(
        ["azim", "elev", "dist", "label", "view_nb", "exp_id"])
    t = tqdm(enumerate(data_loader), total=len(data_loader))
    for i, (targets, meshes, points) in t:

        with torch.no_grad():

            c_batch_size = targets.shape[0]

            azim, elev, dist = models_bag["mvtn"](
                points, c_batch_size=c_batch_size)
            rendered_images, _ = models_bag["mvrenderer"](
                meshes, points,  azim=azim, elev=elev, dist=dist)
            targets = targets.cuda()
            targets = Variable(targets)

            outputs, feat = models_bag["mvnetwork"](rendered_images)
            if retrieval:
                feat = feat.cpu().numpy()
                feat = lfda.transform(feat)
                d, idx_closest = kdtree.query(feat, k=len(features_training))

                for i_query_batch in range(feat.shape[0]):

                    positives = targets_training[idx_closest[i_query_batch, :]
                                                 ] == targets[i_query_batch].cpu().numpy()

                    num = np.cumsum(positives)
                    num[~positives] = 0

                    den = np.array(
                        [i+1 for i in range(len(features_training))])

                    GTP = np.sum(positives)

                    AP = np.sum(num/den)/GTP
                    all_APs.append(AP)

            loss = criterion(outputs, targets)
            c_views = ListDict({"azim": azim.cpu().numpy().reshape(-1).tolist(), "elev": elev.cpu().numpy().reshape(-1).tolist(),
                                "dist": dist.cpu().numpy().reshape(-1).tolist(), "label": np.repeat(targets.cpu().numpy(), setup["nb_views"]).tolist(),
                                "view_nb": int(targets.cpu().numpy().shape[0]) * list(range(setup["nb_views"])),
                                "exp_id": int(targets.cpu().numpy().shape[0]) * int(setup["nb_views"]) * [setup["exp_id"]]})
            views_record.extend(c_views)
            total_loss += loss.item()
            n += 1
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted.cpu() == targets.cpu()).sum()

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

    print("compute training metrics and store training features")

    total = 0.0
    correct = 0.0
    total_loss = 0.0
    n = 0
    feat_list = []
    target_list = []
    views_record = ListDict(
        ["azim", "elev", "dist", "label", "view_nb", "exp_id"])
    t = tqdm(enumerate(data_loader), total=len(data_loader))
    for i, (targets, meshes, points) in t:
        with torch.no_grad():

            c_batch_size = targets.shape[0]
            azim, elev, dist = models_bag["mvtn"](
                points, c_batch_size=c_batch_size)

            rendered_images, _ = models_bag["mvrenderer"](
                meshes, points,  azim=azim, elev=elev, dist=dist)
            targets = targets.cuda()
            targets = Variable(targets)
            outputs, feat = models_bag["mvnetwork"](rendered_images)

            feat_list.append(feat.cpu().numpy())
            target_list.append(targets.cpu().numpy())

            loss = criterion(outputs, targets)
            c_views = ListDict({"azim": azim.cpu().numpy().reshape(-1).tolist(), "elev": elev.cpu().numpy().reshape(-1).tolist(),
                                "dist": dist.cpu().numpy().reshape(-1).tolist(), "label": np.repeat(targets.cpu().numpy(), setup["nb_views"]).tolist(),
                                "view_nb": int(targets.cpu().numpy().shape[0]) * list(range(setup["nb_views"])),
                                "exp_id": int(targets.cpu().numpy().shape[0]) * int(setup["nb_views"]) * [setup["exp_id"]]})
            views_record.extend(c_views)
            total_loss += loss.item()
            n += 1
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted.cpu() == targets.cpu()).sum()
            t.set_description(
                f"{i} - Acc {100 * correct / total :2.2f} - Loss {total_loss / n:2.6f}")
    features = np.concatenate(feat_list)
    targets = np.concatenate(target_list)
    avg_test_acc = 100 * correct / total
    avg_loss = total_loss / n
    return features, targets

def evluate_rotation_robustness(data_loader, models_bag,  setup, max_degs=180.0,):

    total = 0.0
    correct = 0.0

    total_loss = 0.0
    n = 0
    for i, (targets, meshes, points) in enumerate(tqdm(data_loader)):
        with torch.no_grad():

            c_batch_size = targets.shape[0]
            rot_axis = [0.0, 1.0, 0.0]
            angles = [np.random.rand()*20.*max_degs -
                      max_degs for _ in range(c_batch_size)]

            rotR = np.array([rotation_matrix(rot_axis, angle)
                             for angle in angles])
            meshes = Meshes(
                verts=[torch.mm(torch.from_numpy(rotR[ii]).to(torch.float), msh.verts_list()[
                                0].transpose(0, 1)).transpose(0, 1).cuda() for ii, msh in enumerate(meshes)],
                faces=[msh.faces_list()[0].cuda() for msh in meshes],
                textures=None)
            max_vert = meshes.verts_padded().shape[1]

            meshes.textures = Textures(verts_rgb=torch.ones(
                (c_batch_size, max_vert, 3)) .cuda())

            points = torch.bmm(torch.from_numpy(
                rotR).to(torch.float), points.transpose(1, 2)).transpose(1, 2)

            azim, elev, dist = models_bag["mvtn"](
                points, c_batch_size=c_batch_size)
            rendered_images, _ = models_bag["mvrenderer"](
                meshes, points,  azim=azim, elev=elev, dist=dist)
            targets = targets.cuda()
            targets = Variable(targets)
            outputs = models_bag["mvnetwork"](rendered_images)[0]
            loss = criterion(outputs, targets)

            total_loss += loss.item()
            n += 1
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted.cpu() == targets.cpu()).sum()

    avg_test_acc = 100 * correct / total
    avg_loss = total_loss / n

    return avg_test_acc, avg_loss


def view_gcn_exp(setup, models_bag, train_loader, val_loader, dset_val):
    seed_torch()

    models_bag["mvnetwork"].train()
    models_bag["mvtn"].train()
    models_bag["mvrenderer"].train()

    trainer = ModelNetTrainer_mvt(models_bag, train_loader, val_loader, dset_val, nn.CrossEntropyLoss(
    ), 'svcnn', setup["checkpoint_dir1"], num_views=1, setup=setup, classes=classes)

    if setup["resume_first"]:
        trainer.model.load(trainer.weights_dir,)
    if setup["viewgcn_phase"] == "all" or setup["viewgcn_phase"] == "first":
        if setup["run_mode"] == "train":
            trainer.train(setup["first_stage_epochs"])
        else:
            trainer.visualize_views("test", [55, 66, 77])
            trainer.update_validation_accuracy(1)

    models_bag["mvnetwork"] = view_GCN(setup["exp_id"], models_bag["mvnetwork"], nclasses=len(classes),
                                       cnn_name=setup["cnn_name"], num_views=setup["nb_views"])
    models_bag["optimizer"] = torch.optim.SGD(models_bag["mvnetwork"].parameters(), lr=setup["learning_rate"],
                                              weight_decay=setup["weight_decay"], momentum=0.9)

    trainer = ModelNetTrainer_mvt(models_bag, train_loader, val_loader, dset_val,
                                  nn.CrossEntropyLoss(), 'view-gcn', setup["checkpoint_dir2"], num_views=setup["nb_views"], setup=setup, classes=classes)

    if setup["resume"] or "test" in setup["run_mode"]:
        trainer.model.load(trainer.weights_dir,)
        if setup["is_learning_views"]:
            models_bag["mvtn"].load_mvtn(setup["weights_file2"])
    if setup["viewgcn_phase"] == "all" or setup["viewgcn_phase"] == "second":
        if setup["run_mode"] == "train":
            trainer.train(setup["epochs"])
        if setup["run_mode"] == "test_cls":
            trainer.visualize_views("test", all_imgs_list)
            trainer.update_validation_accuracy(1)
        if setup["run_mode"] == "test_retr":
            trainer.train_loader = DataLoader(dset_train, batch_size=int(setup["batch_size"]/2),
                                              shuffle=False, num_workers=6, collate_fn=collate_fn, drop_last=True)
            trainer.update_retrieval()
        if setup["run_mode"] == "test_occ":
            trainer.update_occlusion_robustness()
        if setup["run_mode"] == "test_rot":
            trainer.update_rotation_robustness()
    if setup["log_metrics"]:
        trainer.writer.close()


if setup["resume"] or "test" in setup["run_mode"]:
    if setup["mvnetwork"] in ["mvcnn", "rotnet"]:
        load_checkpoint(setup, models_bag, setup["weights_file"])

if setup["mvnetwork"] == "mvcnn":
    if setup["run_mode"] == "train":
        if setup["log_metrics"]:
            writer = SummaryWriter(setup["logs_dir"])
        for epoch in range(setup["start_epoch"], n_epochs):
            setup["c_epoch"] = epoch
            print('\n-----------------------------------')
            print('Epoch: [%d/%d]' % (epoch+1, n_epochs))
            start = time.time()
            models_bag["mvnetwork"].train()
            models_bag["mvtn"].train()
            models_bag["mvrenderer"].train()

            avg_train_acc, avg_train_loss = train(
                train_loader, models_bag, setup)
            print('Time taken: %.2f sec.' % (time.time() - start))

            models_bag["mvnetwork"].eval()
            models_bag["mvtn"].eval()
            models_bag["mvrenderer"].eval()

            avg_test_acc, avg_loss, views_record = evluate(
                val_loader, models_bag, setup)

            print('\nEvaluation:')
            print('\ttrain acc: %.2f - train Loss: %.4f' %
                  (avg_train_acc.item(), avg_train_loss.item()))
            print('\tVal Acc: %.2f - val Loss: %.4f' %
                  (avg_test_acc.item(), avg_loss))
            print('\tCurrent best val acc: %.2f' % setup["best_acc"])
            if setup["log_metrics"]:
                writer.add_scalar('Loss/train', avg_train_loss.item(), epoch)
                writer.add_scalar('Loss/val', avg_loss, epoch)
                writer.add_scalar('Accuracy/train',
                                  avg_train_acc.item(), epoch)
                writer.add_scalar('Accuracy/val', avg_test_acc.item(), epoch)
            saveables = {'epoch': epoch + 1,
                         'state_dict': models_bag["mvnetwork"].state_dict(),
                         "mvtn": models_bag["mvtn"].state_dict(),

                         'acc': avg_test_acc,
                         'best_acc': setup["best_acc"],
                         'optimizer': models_bag["optimizer"].state_dict(),
                         'mvtn_optimizer': None if not setup["is_learning_views"] else models_bag["mvtn_optimizer"].state_dict(),

                         }
            if setup["save_all"]:
                save_checkpoint(saveables, setup, views_record,
                                setup["weights_file"])

            if avg_test_acc.item() >= setup["best_acc"]:
                print('\tSaving checkpoint - Acc: %.2f' % avg_test_acc)
                saveables["best_acc"] = avg_test_acc
                setup["best_loss"] = avg_loss
                setup["best_acc"] = avg_test_acc.item()
                save_checkpoint(saveables, setup, views_record,
                                setup["weights_file"])

            if (epoch + 1) % setup["lr_decay_freq"] == 0:
                lr *= setup["lr_decay"]
                models_bag["optimizer"] = torch.optim.AdamW(
                    models_bag["mvnetwork"].parameters(), lr=lr)
                print('Learning rate:', lr)
            if (epoch + 1) % setup["plot_freq"] == 0:
                for indx, ii in enumerate(PLOT_SAMPLE_NBS):
                    c_batch_size = 1
                    (targets, meshes, points) = dset_val[ii]
                    cameras_root_folder = os.path.join(
                        setup["cameras_dir"], str(indx))
                    check_folder(cameras_root_folder)
                    renderings_root_folder = os.path.join(
                        setup["renderings_dir"], str(indx))
                    check_folder(renderings_root_folder)
                    cameras_path = os.path.join(
                        cameras_root_folder, "MV_cameras_{}.jpg".format(str(epoch + 1)))
                    images_path = os.path.join(
                        renderings_root_folder, "MV_renderings_{}.jpg".format(str(epoch + 1)))

                    if not setup["return_points_saved"] and not setup["return_points_sampled"]:
                        points = torch.from_numpy(points)
                    azim, elev, dist = models_bag["mvtn"](
                        points[None, ...], c_batch_size=c_batch_size)
                    models_bag["mvrenderer"].render_and_save(
                        [meshes], points[None, ...], azim=azim, elev=elev, dist=dist, images_path=images_path, cameras_path=cameras_path,)
        if setup["log_metrics"]:
            writer.add_hparams(setup, {"hparams/best_acc": setup["best_acc"]})
    if setup["run_mode"] == "test_cls":
        print('\nEvaluation:')
        models_bag["mvnetwork"].eval()
        models_bag["mvtn"].eval()
        models_bag["mvrenderer"].eval()

        avg_test_acc, avg_test_loss, _ = evluate(val_loader, models_bag, setup)
        print('\tVal Acc: %.2f - val Loss: %.4f' %
              (avg_test_acc.item(), avg_test_loss.item()))
        print('\tCurrent best val acc: %.2f' % setup["best_acc"])
        for indx, ii in enumerate(PLOT_SAMPLE_NBS):
            (targets, meshes, points) = dset_val[ii]
            c_batch_size = 1
            cameras_root_folder = os.path.join(setup["cameras_dir"], str(indx))
            check_folder(cameras_root_folder)
            renderings_root_folder = os.path.join(
                setup["renderings_dir"], str(indx))
            check_folder(renderings_root_folder)
            cameras_path = os.path.join(cameras_root_folder,
                                        "MV_cameras_{}.jpg".format("test"))
            images_path = os.path.join(
                renderings_root_folder, "MV_renderings_{}.jpg".format("test"))
            if not setup["return_points_saved"] and not setup["return_points_sampled"]:
                points = torch.from_numpy(points)
            azim, elev, dist = models_bag["mvtn"](
                points[None, ...], c_batch_size=c_batch_size)
            models_bag["mvrenderer"].render_and_save(
                [meshes], points[None, ...], azim=azim, elev=elev, dist=dist, images_path=images_path, cameras_path=cameras_path,)
    if setup["run_mode"] == "test_retr":
        print('\nEvaluation:')
        models_bag["mvnetwork"].eval()
        models_bag["mvtn"].eval()
        models_bag["mvrenderer"].eval()

        os.makedirs(os.path.dirname(setup["feature_file"]), exist_ok=True)
        if not os.path.exists(setup["feature_file"]) or not os.path.exists(setup["targets_file"]):
            features, targets = compute_features(
                train_loader, models_bag, setup)
            np.save(setup["feature_file"], features)
            np.save(setup["targets_file"], targets)

        LFDA_reduction_file = os.path.join(
            setup["features_dir"], "reduction_LFDA.pkl")
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

        avg_test_acc, avg_test_retr_mAP, avg_test_loss, _ = evluate(
            val_loader, models_bag, setup, retrieval=True)
        print('\tVal Acc: %.2f - val retr-mAP: %.2f - val Loss: %.4f' %
              (avg_test_acc.item(), avg_test_retr_mAP, avg_test_loss.item()))
        print('\tCurrent best val acc: %.2f' % setup["best_acc"])

    elif setup["run_mode"] == "test_occ":
        models_bag["mvnetwork"].eval()
        models_bag["mvtn"].eval()
        models_bag["mvrenderer"].eval()

        if "modelnet" not in setup["data_dir"].lower():
            raise Exception('Occlusion is only supported froom ModelNet now ')
        from tqdm import tqdm
        torch.multiprocessing.set_sharing_strategy('file_system')

        print('\Evaluatiing om the cropped data :')

        override = True
        networks_list = ["MVTN"]

        factor_list = [-0.75, -0.5, -0.3, -
                       0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.5, 0.75]
        axis_list = [0, 1, 2]

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
            if network in ["DGCNN", "PointNet"]:
                point_network.eval()
                load_point_ckpt(
                    point_network,  setup["shape_extractor"],  ckpt_dir='./checkpoint')
            exp_id = "chopping_{}".format(network)
            save_file = os.path.join(setup["results_dir"], exp_id+".csv")
            if not os.path.isfile(save_file) or override:
                t = tqdm(enumerate(val_loader), total=len(val_loader))
                for ii, (targets, meshes, orig_pts) in t:
                    c_batch_size = len(meshes)
                    with torch.no_grad():
                        azim, elev, dist = models_bag["mvtn"](
                            points, c_batch_size=c_batch_size)
                        rendered_images, _ = models_bag["mvrenderer"](
                            meshes, points,  azim=azim, elev=elev, dist=dist)
                        targets = targets.cuda()
                        for factor in factor_list:
                            for axis in axis_list:
                                c_setup = {"network": network,
                                           "batch": ii, "factor": factor, "axis": axis}
                                [setups.append(c_setup)
                                 for ii in range(c_batch_size)]
                                chopped_pts = chop_ptc(
                                    orig_pts.cpu().numpy(), factor, axis=axis)
                                chopped_pts = torch.from_numpy(chopped_pts)
                                if network not in ["PointNet", "DGCNN"]:
                                    azim, elev, dist = models_bag["mvtn"](
                                        points, c_batch_size=c_batch_size)
                                    rendered_images, _ = models_bag["mvrenderer"](
                                        meshes, points,  azim=azim, elev=elev, dist=dist)
                                    outputs, _ = models_bag["mvnetwork"](
                                        rendered_images)
                                else:
                                    chopped_pts = chopped_pts.transpose(
                                        1, 2).cuda()
                                    outputs = point_network(chopped_pts)[
                                        0].view(c_batch_size, -1)
                                _, predictions = torch.max(outputs.data, 1)
                                c_result = ListDict({"prediction": predictions.cpu().numpy(
                                ).tolist(), "class": targets.cpu().numpy().tolist()})
                                results.extend(c_result)
                                save_results(save_file, results+setups)

    elif setup["run_mode"] == "test_rot":
        setup["results_file"] = os.path.join(
            setup["results_dir"], setup["exp_id"]+"_robustness_{}.csv".format(str(int(setup["max_degs"]))))
        setup["return_points_saved"] = True
        assert os.path.isfile(setup["weights_file"]
                              ), 'Error: no checkpoint file found!'

        loaded_info = load_results(os.path.join(
            setup["results_dir"], setup["exp_id"]+"_accuracy.csv"))
        setup["start_epoch"] = loaded_info["start_epoch"][0]
        setup["nb_views"] = loaded_info["nb_views"][0]
        setup["views_config"] = loaded_info["views_config"][0]

        print('\nEvaluating Robustness:')

        mvtn = MVTN(setup["nb_views"], views_config=setup["views_config"],
                    canonical_elevation=setup["canonical_elevation"], canonical_distance=setup["canonical_distance"],
                    shape_features_size=setup["features_size"], transform_distance=setup["transform_distance"], input_view_noise=setup["input_view_noise"], shape_extractor=setup["shape_extractor"], screatch_feature_extractor=setup["screatch_feature_extractor"]).cuda()
        models_bag["mvtn"] = mvtn

        load_checkpoint_robustness(setup, models_bag, setup["weights_file"])
        models_bag["mvnetwork"].eval()
        models_bag["mvtn"].eval()
        models_bag["mvrenderer"].eval()

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
        raise ValueError(
            "batch size should be multiplication of the number of views")
    vcand = np.load('rotationNet/vcand_case1.npy')
    if setup["log_metrics"]:
        writer = SummaryWriter(setup["logs_dir"])
    for epoch in range(setup["start_epoch"], n_epochs):
        setup["c_epoch"] = epoch
        print('\n-----------------------------------')
        print('Epoch: [%d/%d]' % (epoch+1, n_epochs))
        if setup["run_mode"] == "train":
            start = time.time()
            models_bag["mvnetwork"].train()
            models_bag["mvtn"].train()
            models_bag["mvrenderer"].train()

            avg_train_acc, avg_train_loss = train_rotationNet(
                train_loader, models_bag, setup)
            print('Time taken: %.2f sec.' % (time.time() - start))
            print('\ttrain acc: %.2f - train Loss: %.4f' %
                  (avg_train_acc, avg_train_loss.item()))
            models_bag["mvnetwork"].eval()
            models_bag["mvtn"].eval()
            models_bag["mvrenderer"].eval()

            avg_test_acc, avg_loss = evaluate_rotationNet(
                val_loader, models_bag, setup)

        print('\nEvaluation:')

        print('\tVal Acc: %.2f - val Loss: %.4f' %
              (avg_test_acc, avg_loss))
        print('\tCurrent best val acc: %.2f' % setup["best_acc"])
        if setup["log_metrics"] and setup["run_mode"] == "train":
            writer.add_scalar('Loss/train', avg_train_loss.item(), epoch)
            writer.add_scalar('Loss/val', avg_loss, epoch)
            writer.add_scalar('Accuracy/train', avg_train_acc, epoch)
            writer.add_scalar('Accuracy/val', avg_test_acc, epoch)

        saveables = {'epoch': epoch + 1,
                     'state_dict': models_bag["mvnetwork"].state_dict(),
                     "mvtn": models_bag["mvtn"].state_dict(),

                     'acc': avg_test_acc,
                     'best_acc': setup["best_acc"],
                     'optimizer': models_bag["optimizer"].state_dict(),
                     'mvtn_optimizer': None if not setup["is_learning_views"] else models_bag["mvtn_optimizer"].state_dict(),

                     }

        if avg_test_acc >= setup["best_acc"]:
            print('\tSaving checkpoint - Acc: %.2f' % avg_test_acc)
            saveables["best_acc"] = avg_test_acc
            setup["best_loss"] = avg_loss
            setup["best_acc"] = avg_test_acc
            save_checkpoint(saveables, setup, None,
                            setup["weights_file"])

        if (epoch + 1) % setup["lr_decay_freq"] == 0:
            lr *= setup["lr_decay"]
            models_bag["optimizer"] = torch.optim.AdamW(
                models_bag["mvnetwork"].parameters(), lr=lr)
            print('Learning rate:', lr)
    if setup["log_metrics"] and setup["run_mode"] == "train":
        writer.add_hparams(setup, {"hparams/best_acc": setup["best_acc"]})

elif setup["mvnetwork"] == "viewgcn":
    if setup["resume_mvtn"]:
        models_bag["mvtn"].load_mvtn(setup["weights_file2"])
        setup["mvtn_learning_rate"] = 0.0
        setup["pn_learning_rate"] = 0.0

    all_imgs_list = [55, 66, 77]
    view_gcn_exp(setup, models_bag, train_loader, val_loader, dset_val)
