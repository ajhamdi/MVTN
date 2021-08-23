import torch
from torch.autograd import Variable
import numpy as np
import os
import sys
from util import *
import shutil
from torch import nn
from torch._six import inf
from einops import rearrange, repeat
from einops.layers.torch import Rearrange


from pytorch3d.structures import Meshes, Pointclouds
from pytorch3d.renderer.mesh import Textures
from pytorch3d.renderer import (
    OpenGLPerspectiveCameras, look_at_view_transform, look_at_rotation,
    RasterizationSettings, MeshRenderer, MeshRasterizer, BlendParams,
    SoftSilhouetteShader, HardPhongShader, PointLights, SoftPhongShader,
    HardFlatShader, HardGouraudShader, SoftGouraudShader,
    OpenGLOrthographicCameras,
    PointsRasterizationSettings,
    PointsRenderer,
    PointsRasterizer,
    AlphaCompositor,
    NormWeightedCompositor, DirectionalLights)
# ORTHOGONAL_THRESHOLD = 1e-6
EXAHSTION_LIMIT = 20
# EPSILON = 0.00001


def initialize_setup(setup):
    SHAPE_FEATURES_SIZE = {"logits": 40, "post_max": 1024,
                           "transform_matrix": 64*64, "pre_linear": 512, "post_max_trans": 1024 + 64*64, "logits_trans": 40+64*64, "pre_linear_trans": 512+64*64}

    setup["features_size"] = SHAPE_FEATURES_SIZE[setup["features_type"]]
    if setup["exp_id"] == "random":
        setup["exp_id"] = random_id()
    check_folder(os.path.join(setup["results_dir"], setup["exp_set"]))
    setup["results_dir"] = os.path.join(
        setup["results_dir"], setup["exp_set"], setup["exp_id"])
    setup["cameras_dir"] = os.path.join(
        setup["results_dir"], setup["cameras_dir"])
    setup["renderings_dir"] = os.path.join(
        setup["results_dir"], setup["renderings_dir"])
    setup["verts_dir"] = os.path.join(setup["results_dir"], "verts")
    setup["checkpoint_dir"] = os.path.join(setup["results_dir"], "checkpoint")
    setup["features_dir"] = os.path.join(setup["results_dir"], "features")
    setup["logs_dir"] = os.path.join(setup["results_dir"], setup["logs_dir"])
    setup["feature_file"] = os.path.join(
        setup["features_dir"], "features_training.npy")
    setup["targets_file"] = os.path.join(
        setup["features_dir"], "targets_training.npy")

    check_folder(setup["results_dir"])
    check_folder(setup["cameras_dir"])
    check_folder(setup["renderings_dir"])
    check_folder(setup["logs_dir"])
    check_folder(setup["verts_dir"])
    check_folder(setup["checkpoint_dir"])
    setup["best_acc"] = 0.0
    setup["best_loss"] = 0.0
    setup["start_epoch"] = 0
    setup["results_file"] = os.path.join(
        setup["results_dir"], setup["exp_id"]+"_accuracy.csv")
    setup["views_file"] = os.path.join(
        setup["results_dir"], setup["exp_id"]+"_views.csv")
    setup["weights_file"] = os.path.join(
        setup["checkpoint_dir"], setup["exp_id"]+"_checkpoint.pt")
    setup["is_learning_views"] = setup["views_config"] in ["learned_offset",
                                                             "learned_direct", "learned_spherical", "learned_random", "learned_transfer"]
    setup["is_learning_points"] = setup["is_learning_views"] and (
        setup["return_points_saved"] or setup["return_points_sampled"])
    for k, v in setup.items():
        if isinstance(v, bool):
            setup[k] = int(v)


def initialize_setup_gcn(setup):
    SHAPE_FEATURES_SIZE = {"logits": 40, "post_max": 1024,
                           "transform_matrix": 64*64, "pre_linear": 512, "post_max_trans": 1024 + 64*64, "logits_trans": 40+64*64, "pre_linear_trans": 512+64*64}

    setup["features_size"] = SHAPE_FEATURES_SIZE[setup["features_type"]]
    if setup["exp_id"] == "random":
        setup["exp_id"] = random_id()
    # setup["results_dir"] = os.path.join(setup["GCN_dir"], setup["results_dir"])
    r_dir = copy.deepcopy(setup["results_dir"])
    check_folder(os.path.join(setup["results_dir"], setup["exp_set"]))
    setup["results_dir"] = os.path.join(
        setup["results_dir"], setup["exp_set"], setup["exp_id"])
    setup["cameras_dir"] = os.path.join(
        setup["results_dir"], setup["cameras_dir"])
    setup["renderings_dir"] = os.path.join(
        setup["results_dir"], setup["renderings_dir"])
    setup["verts_dir"] = os.path.join(setup["results_dir"], "verts")
    setup["checkpoint_dir1"] = os.path.join(
        setup["results_dir"], "checkpoint_stage1")
    setup["checkpoint_dir2"] = os.path.join(
        setup["results_dir"], "checkpoint_stage2")
    # setup["train_path"] = os.path.join(setup["GCN_dir"], setup["train_path"])
    # setup["val_path"] = os.path.join(setup["GCN_dir"], setup["val_path"])
    setup["cnn_name"] = "resnet{}".format(setup["depth"])
    setup["logs_dir"] = os.path.join(setup["results_dir"], setup["logs_dir"])

    setup["features_dir"] = os.path.join(setup["results_dir"], "features")
    setup["feature_file"] = os.path.join(
        setup["features_dir"], f"features_layer{setup['LFDA_layer']}_training.npy")
    setup["targets_file"] = os.path.join(
        setup["features_dir"], f"targets_layer{setup['LFDA_layer']}_training.npy")
    setup["LFDA_file"] = os.path.join(
        setup["features_dir"], f"reduction_LFDA_{setup['LFDA_dimension']}_layer{setup['LFDA_layer']}.pkl")

    check_folder(setup["results_dir"])
    check_folder(setup["cameras_dir"])
    check_folder(setup["renderings_dir"])
    check_folder(setup["logs_dir"])
    check_folder(setup["verts_dir"])
    check_folder(setup["checkpoint_dir1"])
    check_folder(setup["checkpoint_dir2"])
    if setup["run_mode"] != "test_cls" and setup["resume_first"]:
        # if not setup["normalize_properly"]:
            # shutil.copyfile(os.path.join(r_dir, "model-00029.pth"),
            #         os.path.join(setup["checkpoint_dir1"], "model-00029.pth"))
        if "modelnet" in setup["data_dir"].lower():
            dset_name = "modelnet"
            ckpt_nb = 29
        elif "shapenet" in setup["data_dir"].lower():
            dset_name = "shapenet"
            ckpt_nb = 29
        elif "scanobjectnn" in setup["data_dir"].lower():
            dset_name = "scanobjectnn"
            ckpt_nb = 29

        shutil.copyfile(os.path.join(r_dir, "checkpoints", dset_name, "model-000{}.pth".format(ckpt_nb)),
                        os.path.join(setup["checkpoint_dir1"], "model-000{}.pth".format(ckpt_nb)))
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
    setup["is_learning_views"] = setup["views_config"] in [
        "learned_offset", "learned_direct", "learned_spherical", "learned_random", "learned_transfer"]
    setup["is_learning_points"] = setup["is_learning_views"] and (
        setup["return_points_saved"] or setup["return_points_sampled"])
    for k, v in setup.items():
        if isinstance(v, bool):
            setup[k] = int(v)



# MVTN_regressor = Sequential(MLP([b+2*M, b, b, 5 * M, 2*M], activation="relu", dropout=0.5,
                                # batch_norm=True), MLP([2*M, 2*M], activation=None, dropout=0, batch_norm=False), nn.Tanh())
def applied_transforms(images_batch, crop_ratio=0.3):
    """
    a pyutroch transforms that can be applied batchwise 
    """
    N, C, H, W = images_batch.shape
    padd = torch.nn.ReplicationPad2d(int((1+crop_ratio)*H)-H)
    images_batch = RandomHorizontalFlip()(images_batch)
    images_batch = RandomCrop(H)(padd(images_batch))
    return images_batch


def super_batched_op(dim,batched_ops,batched_tensor,*args,**kwargs):
    """
    convert a batch operation in pytorch to work on 5 dims (N,C,H,W) + X , where `dim` will dictate the extra dimension X that will be put on dimensions N  
    """
    return unbatch_tensor(batched_ops(batch_tensor(batched_tensor, dim=dim, squeeze=True), *args, **kwargs), dim=dim, unsqueeze=True, batch_size=batched_tensor.shape[0])


def check_and_correct_rotation_matrix(R, T, nb_trials, azim, elev, dist):
    exhastion = 0
    while not check_valid_rotation_matrix(R):
            exhastion += 1
            R, T = look_at_view_transform(dist=batch_tensor(dist.T, dim=1, squeeze=True), elev=batch_tensor(elev.T + 90.0 * torch.rand_like(elev.T, device=elev.device),
                                                                                                                                 dim=1, squeeze=True), azim=batch_tensor(azim.T + 180.0 * torch.rand_like(azim.T, device=elev.device), dim=1, squeeze=True))
            # print("PROBLEM is fixed {} ? : ".format(exhastion),check_valid_rotation_matrix(R))
            if not check_valid_rotation_matrix(R) and exhastion > nb_trials:
                sys.exit("Remedy did not work")
    return R , T

# def render_meshes(meshes, color, azim, elev, dist, lights, setup, background_color=(1.0, 1.0, 1.0), ):
#     c_batch_size = len(meshes)
#     verts = [msh.verts_list()[0].cuda() for msh in meshes]
#     faces = [msh.faces_list()[0].cuda() for msh in meshes]
#     # faces = [torch.cat((fs, torch.flip(fs, dims=[1])),dim=0) for fs in faces]
#     new_meshes = Meshes(
#         verts=verts,
#         faces=faces,
#         textures=None)
#     max_vert = new_meshes.verts_padded().shape[1]

#     # print(len(new_meshes.faces_list()[0]))
#     new_meshes.textures = Textures(
#         verts_rgb=color.cuda()*torch.ones((c_batch_size, max_vert, 3)).cuda())
#     # Create a Meshes object for the teapot. Here we have only one mesh in the batch.
#     R, T = look_at_view_transform(dist=batch_tensor(dist.T, dim=1, squeeze=True), elev=batch_tensor(
#         elev.T, dim=1, squeeze=True), azim=batch_tensor(azim.T, dim=1, squeeze=True))
#     R, T = check_and_correct_rotation_matrix(R, T, EXAHSTION_LIMIT, azim, elev, dist)

#     cameras = OpenGLPerspectiveCameras(device="cuda:{}".format(torch.cuda.current_device()), R=R, T=T)
#     camera = OpenGLPerspectiveCameras(device="cuda:{}".format(torch.cuda.current_device()), R=R[None, 0, ...],
#                                       T=T[None, 0, ...])

#     # camera2 = OpenGLPerspectiveCameras(device=device, R=R[None, 2, ...],T=T[None, 2, ...])
#     # print(camera2.get_camera_center())
#     raster_settings = RasterizationSettings(
#         image_size=setup["image_size"],
#         blur_radius=0.0,
#         faces_per_pixel=1,
#         # bin_size=None, #int
#         # max_faces_per_bin=None,  # int
#         # perspective_correct=False,
#         # clip_barycentric_coords=None, #bool
#         cull_backfaces=setup["cull_backfaces"],
#     )
#     renderer = MeshRenderer(
#         rasterizer=MeshRasterizer(
#             cameras=camera, raster_settings=raster_settings),
#         shader=HardPhongShader(blend_params=BlendParams(background_color=background_color
#                                                         ), device=lights.device, cameras=camera, lights=lights)
#     )
#     new_meshes = new_meshes.extend(setup["nb_views"])

#     # compute output
#     # print("after rendering .. ", rendered_images.shape)

#     rendered_images = renderer(new_meshes, cameras=cameras, lights=lights)

#     rendered_images = unbatch_tensor(
#         rendered_images, batch_size=setup["nb_views"], dim=1, unsqueeze=True).transpose(0, 1)
#     # print(rendered_images[:, 100, 100, 0])

#     rendered_images = rendered_images[..., 0:3].transpose(2, 4).transpose(3, 4)
#     return rendered_images, cameras


# def render_points(points, color, azim, elev, dist, setup, background_color=(0.0, 0.0, 0.0), ):
#     c_batch_size = azim.shape[0]

#     point_cloud = Pointclouds(points=points.to(torch.float), features=color *
#                               torch.ones_like(points, dtype=torch.float)).cuda()

#     # print(len(new_meshes.faces_list()[0]))
#     # Create a Meshes object for the teapot. Here we have only one mesh in the batch.
#     R, T = look_at_view_transform(dist=batch_tensor(dist.T, dim=1, squeeze=True), elev=batch_tensor(
#         elev.T, dim=1, squeeze=True), azim=batch_tensor(azim.T, dim=1, squeeze=True))
#     R, T = check_and_correct_rotation_matrix(R, T, EXAHSTION_LIMIT, azim, elev, dist)

#     cameras = OpenGLOrthographicCameras(device="cuda:{}".format(torch.cuda.current_device()), R=R, T=T, znear=0.01)
#     raster_settings = PointsRasterizationSettings(
#         image_size=setup["image_size"],
#         radius=setup["points_radius"],
#         points_per_pixel=setup["points_per_pixel"]
#     )

#     renderer = PointsRenderer(
#         rasterizer=PointsRasterizer(
#             cameras=cameras, raster_settings=raster_settings),
#         compositor=NormWeightedCompositor()
#     )
#     point_cloud = point_cloud.extend(setup["nb_views"])
#     point_cloud.scale_(batch_tensor(1.0/dist.T, dim=1,squeeze=True)[..., None][..., None])

#     rendered_images = renderer(point_cloud)
#     rendered_images = unbatch_tensor(
#         rendered_images, batch_size=setup["nb_views"], dim=1, unsqueeze=True).transpose(0, 1)

#     rendered_images = rendered_images[..., 0:3].transpose(2, 4).transpose(3, 4)
#     return rendered_images, cameras


# def auto_render_meshes(targets, meshes, points, models_bag, setup, ):
#     # inputs = inputs.cuda(device)
#     # inputs = Variable(inputs)
#     c_batch_size = len(targets)
#     # if the model in test phase use white color
#     if setup["object_color"] == "random" and not models_bag["mvtn"].training:
#         color = torch_color("white")
#     else:
#         color = torch_color(setup["object_color"],max_lightness=True, epsilon=EPSILON)
#     background_color = torch_color(setup["background_color"], max_lightness=True, epsilon=EPSILON).cuda()
#     # shape_features = models_bag["feature_extractor"](
#     #     points, c_batch_size=c_batch_size).cuda()
#     azim, elev, dist = models_bag["mvtn"](points, c_batch_size=c_batch_size)

#     # lights = PointLights(
#     #     device=None, location=((0, 0, 0),))
#     if not setup["pc_rendering"]:
#         lights = DirectionalLights(
#             device=background_color.device, direction=models_bag["mvtn"].light_direction(azim, elev, dist))

#         rendered_images, cameras = render_meshes(
#             meshes=meshes, color=color, azim=azim, elev=elev, dist=dist, lights=lights, setup=setup, background_color=background_color)
#     else:
#         rendered_images, cameras = render_points(
#             points=points, color=color, azim=azim, elev=elev, dist=dist, setup=setup, background_color=background_color)
#     return rendered_images, cameras, azim, elev, dist




# def auto_render_meshes_custom_views(targets, meshes, points, models_bag, setup, ):
#     c_batch_size = len(targets)
#     if setup["object_color"] == "random" and not models_bag["mvtn"].training:
#         color = torch_color("white")
#     else:
#         color = torch_color(setup["object_color"],max_lightness=True)

#     # shape_features = models_bag["feature_extractor"](
#     #     points, c_batch_size=c_batch_size).cuda()
#     azim, elev, dist = models_bag["mvtn"](points, batch_size=c_batch_size)

#     for i, target in enumerate(targets.numpy().tolist()):
#         azim[i] = torch.from_numpy(np.array(models_bag["azim_dict"][target]))
#         elev[i] = torch.from_numpy(np.array(models_bag["elev_dict"][target]))

#     # lights = PointLights(
#     #     device=device, location=((0, 0, 0),))
#     if not setup["pc_rendering"]:
#         lights = DirectionalLights(
#             targets.device, direction=models_bag["mvtn"].light_direction(azim, elev, dist))

#         rendered_images, cameras = render_meshes(
#             meshes=meshes, color=color, azim=azim, elev=elev, dist=dist, lights=lights, setup=setup,)
#     else:
#         rendered_images, cameras = render_points(
#             points=points, azim=azim, elev=elev, dist=dist, setup=setup,)
#     rendered_images = nn.functional.dropout2d(
#         rendered_images, p=setup["view_reg"], training=models_bag["mvtn"].training)

#     return rendered_images, cameras, azim, elev, dist


# def auto_render_and_save_images_and_cameras(targets, meshes, points, images_path, cameras_path, models_bag, setup, ):
#     # inputs = np.stack(inputs, axis=0)
#     # inputs = torch.from_numpy(inputs)
#     with torch.no_grad():
#         if not setup["return_points_saved"] and not setup["return_points_sampled"]:
#             points = torch.from_numpy(points)
#         targets = torch.tensor(targets)[None]
#         # correction_factor = torch.tensor(correction_factor)
#         rendered_images, cameras, _, _, _ = auto_render_meshes(
#             targets, [meshes], points[None, ...], models_bag, setup,)
#     # print("before saving .. ",rendered_images.shape)
#     save_grid(image_batch=rendered_images[0, ...],
#               save_path=images_path, nrow=setup["nb_views"])
#     save_cameras(cameras, save_path=cameras_path, scale=0.22, dpi=200)


# def auto_render_and_analyze_images(targets, meshes, points, images_path, models_bag, setup, ):
#     # inputs = np.stack(inputs, axis=0)
#     # inputs = torch.from_numpy(inputs)
#     with torch.no_grad():
#         if not setup["return_points_saved"] and not setup["return_points_sampled"]:
#             points = torch.from_numpy(points)
#         targets = torch.tensor(targets)
#         # correction_factor = torch.tensor(correction_factor)
#         rendered_images, _, _, _, _ = auto_render_meshes(
#             targets, [meshes], points[None, ...], models_bag, setup,)
#     # print("before saving .. ",rendered_images.shape)
#     save_grid(image_batch=rendered_images[0, ...],
#               save_path=images_path, nrow=setup["nb_views"])
#     mask = rendered_images != 1
#     img_avg = (rendered_images*mask).sum(dim=(0, 1, 2, 3, 4)) / \
#         mask.sum(dim=(0, 1, 2, 3, 4))
#     # print("Original  ", rendered_images.mean(dim=(0,1,2,3,4)), "\n IMG avg : ", img_avg)
#     # print(img_avg.cpu().numpy())

#     return float(img_avg.cpu().numpy())


def regualarize_rendered_views(rendered_images, dropout_p=0, augment_training=False, crop_ratio=0.3):
    #### To perform dropout on the views
    rendered_images = nn.functional.dropout2d(
        rendered_images, p=dropout_p, training=True)

    if augment_training:
        rendered_images = super_batched_op(
            1, applied_transforms, rendered_images, crop_ratio=crop_ratio)
    return rendered_images

def clip_grads_(parameters, max_norm, norm_type=2):
    r"""Clips gradient norm of an iterable of parameters and zero them if nan.

    The norm is computed over all gradients together, as if they were
    concatenated into a single vector. Gradients are modified in-place.

    Arguments:
        parameters (Iterable[Tensor] or Tensor): an iterable of Tensors or a
            single Tensor that will have gradients normalized
        max_norm (float or int): max norm of the gradients
        norm_type (float or int): type of the used p-norm. Can be ``'inf'`` for
            infinity norm.

    Returns:
        Total norm of the parameters (viewed as a single vector).
    """
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    max_norm = float(max_norm)
    norm_type = float(norm_type)
    for p in parameters:
        p.grad.detach().data = zero_nans(p.grad.detach().data)
    if norm_type == inf:
        total_norm = max(p.grad.detach().abs().max() for p in parameters)
    else:
        total_norm = torch.norm(torch.stack(
            [torch.norm(p.grad.detach(), norm_type) for p in parameters]), norm_type)
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        for p in parameters:
            p.grad.detach().mul_(clip_coef)
    return total_norm








def test_point_network(model,criterion, data_loader):
    total = 0.0
    correct = 0.0
    total_loss = 0.0
    n = 0
    from tqdm import tqdm
    for i, (targets,_, points,_) in enumerate(tqdm(data_loader)):
        with torch.no_grad():
            points = points.transpose(1, 2).cuda()
            targets = targets.cuda()
            targets = Variable(targets)
            # print(rendered_images[:,0,:,100,100])
            logits, shape_features, trans = model(points)
            loss = criterion(logits, targets)

            total_loss += loss
            n += 1
            _, predicted = torch.max(logits.data, 1)
            total += targets.size(0)
            correct += (predicted.cpu() == targets.cpu()).sum()

    avg_test_acc = 100 * correct / total
    avg_loss = total_loss / n

    return avg_test_acc, avg_loss




def save_checkpoint(state, setup, views_record, weights_file, ignore_saving_models=False):
    if not ignore_saving_models:
        torch.save(state, weights_file)
    setup_dict = ListDict(list(setup.keys()))
    save_results(setup["results_file"], setup_dict.append(setup))
    if views_record is not None:
        save_results(setup["views_file"], views_record)


def load_checkpoint(setup, models_bag, weights_file):
    # Load checkpoint.
    print('\n==> Loading checkpoint..')
    assert os.path.isfile(weights_file
                          ), 'Error: no checkpoint file found!'

    checkpoint = torch.load(weights_file)
    setup["best_acc"] = checkpoint['best_acc']
    setup["start_epoch"] = checkpoint['epoch']
    models_bag["mvnetwork"].load_state_dict(checkpoint['state_dict'])
    if setup["is_learning_views"]:
        models_bag["mvtn"].load_state_dict(
            checkpoint['mvtn'])
        models_bag["mvtn_optimizer"].load_state_dict(checkpoint['mvtn_optimizer'])
    # if setup["is_learning_points"]:
    #     models_bag["feature_extractor"].load_state_dict(
    #         checkpoint['feature_extractor'])
    #     models_bag["fe_optimizer"].load_state_dict(checkpoint['fe_optimizer'])
    # if "late_fusion_mode" in setup and setup["late_fusion_mode"]:
    #     models_bag["classifier"].load_state_dict(checkpoint['classifier'])
    #     models_bag["cls_optimizer"].load_state_dict(checkpoint['cls_optimizer'])
    #     models_bag["point_network"].load_state_dict(
    #         checkpoint['point_network'])
    #     models_bag["fe_optimizer"].load_state_dict(checkpoint['fe_optimizer'])

    models_bag["optimizer"].load_state_dict(checkpoint['optimizer'])





def load_checkpoint_robustness(setup, models_bag, weights_file):
    # Load checkpoint.
    print('\n==> Loading checkpoint..')
    assert os.path.isfile(weights_file
                          ), 'Error: no checkpoint file found!'

    checkpoint = torch.load(weights_file)
    models_bag["mvnetwork"].load_state_dict(checkpoint['state_dict'])
    if setup["is_learning_views"]:
        models_bag["mvtn"].load_state_dict(
            checkpoint['mvtn'])
    # if setup["is_learning_points"]:
    #     models_bag["feature_extractor"].load_state_dict(
    #         checkpoint['feature_extractor'])





def mvtosv(x): return rearrange(x, 'b m h w -> (b m) h w ')
def mvctosvc(x): return rearrange(x, 'b m c h w -> (b m) c h w ')
def svtomv(x,nb_views=1): return rearrange(x, '(b m) h w -> b m h w',m=nb_views)
def svctomvc(x,nb_views=1): return rearrange(x, '(b m) c h w -> b m c h w',m=nb_views)




def extra_IOU_metrics(points_GT, points_predictions, pixels_GT, pixel_mask, points_mask,object_class, parts,):
    """
    a funciton to calculate IOUs  for bacth of point clouds `points_predictions` based on the ground truth `points_GT` and record more metrics as well based on pixels
    """
    bs , p_nb = points_GT.shape
    _, v,h,w = pixels_GT.shape
    cur_shape_ious = []
    cur_parts_valid = []
    part_nb = [] ; cls_nb = []
    pixel_perc = []  ;  point_perc = []
    for cl in range(torch.max(parts).item()):
        cur_gt_mask = (points_GT == cl) & points_mask  # -1 to remove the background class laabel
        cur_pred_mask = (points_predictions == cl) & points_mask

        I = (cur_pred_mask & cur_gt_mask).sum(dim=-1)
        U = (cur_pred_mask | cur_gt_mask).sum(dim=-1)

        cur_shape_ious.extend((100.0* I/(U + 1e-7) ).cpu().numpy().tolist() )
        cur_parts_valid.extend((U > 0).to(torch.int32).cpu().numpy().tolist())
        cls_nb.extend(object_class.squeeze().cpu().numpy().tolist())
        part_nb.extend(bs*[cl])
        pixel_perc.extend((100.0*(pixels_GT == cl).sum(dim=-1).sum(dim=-1).sum(dim=-1).to(
            torch.float).cpu().numpy() / (pixel_mask.sum().item())).tolist())
        point_perc.extend((100.0*cur_gt_mask.sum(dim=-1).to(torch.float).cpu().numpy() / points_mask.sum().item()).tolist())

    return  pixel_perc, point_perc, cur_shape_ious,cur_parts_valid, cls_nb ,part_nb 


