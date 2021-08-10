import torch
from torch.autograd import Variable
import math
import numpy as np
import os
from util import *
import imageio
from models.blocks import *
from models.pointnet import *
import shutil
from torch import nn
from torch._six import inf
from ptflops import get_model_complexity_info
from einops import rearrange, repeat
from einops.layers.torch import Rearrange


from pytorch3d.renderer.cameras import camera_position_from_spherical_angles
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
ORTHOGONAL_THRESHOLD = 1e-6
EXAHSTION_LIMIT = 20
EPSILON = 0.00001


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
    setup["is_learning_views"] = setup["selection_type"] in ["learned_offset",
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
    if not setup["test_only"] and setup["resume_first"]:
        # if not setup["normalize_properly"]:
            # shutil.copyfile(os.path.join(r_dir, "model-00029.pth"),
            #         os.path.join(setup["checkpoint_dir1"], "model-00029.pth"))
        if "modelnet" in setup["mesh_data"].lower():
            dset_name = "modelnet"
            ckpt_nb = 29
        elif "shapenet" in setup["mesh_data"].lower():
            dset_name = "shapenet"
            ckpt_nb = 29
        elif "scanobjectnn" in setup["mesh_data"].lower():
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
    setup["is_learning_views"] = setup["selection_type"] in [
        "learned_offset", "learned_direct", "learned_spherical", "learned_random", "learned_transfer"]
    setup["is_learning_points"] = setup["is_learning_views"] and (
        setup["return_points_saved"] or setup["return_points_sampled"])
    for k, v in setup.items():
        if isinstance(v, bool):
            setup[k] = int(v)



# MVTN_regressor = Sequential(MLP([b+2*M, b, b, 5 * M, 2*M], activation="relu", dropout=0.5,
                                # batch_norm=True), MLP([2*M, 2*M], activation=None, dropout=0, batch_norm=False), nn.Tanh())
def applied_transforms(images_batch, crop_ratio=0.3):
    N, C, H, W = images_batch.shape
    padd = torch.nn.ReplicationPad2d(int((1+crop_ratio)*H)-H)
    images_batch = RandomHorizontalFlip()(images_batch)
    images_batch = RandomCrop(H)(padd(images_batch))
    return images_batch


def super_batched_op(dim,batched_ops,batched_tensor,*args,**kwargs):
    """
    convert a batch operation in pytorch to work on on 5 dims (N,C,H,W) + X , where `dim` will dictate the extra dimension X that will be put on dimensions N  
    """
    return unbatch_tensor(batched_ops(batch_tensor(batched_tensor, dim=dim, squeeze=True), *args, **kwargs), dim=dim, unsqueeze=True, batch_size=batched_tensor.shape[0])


def render_meshes(meshes, color, azim, elev, dist, lights, setup, background_color=(1.0, 1.0, 1.0), device="cuda:0"):
    c_batch = len(meshes)
    verts = [msh.verts_list()[0].cuda() for msh in meshes]
    faces = [msh.faces_list()[0].cuda() for msh in meshes]
    # faces = [torch.cat((fs, torch.flip(fs, dims=[1])),dim=0) for fs in faces]
    new_meshes = Meshes(
        verts=verts,
        faces=faces,
        textures=None)
    max_vert = new_meshes.verts_padded().shape[1]

    # print(len(new_meshes.faces_list()[0]))
    new_meshes.textures = Textures(
        verts_rgb=color.cuda()*torch.ones((c_batch, max_vert, 3)).cuda())
    # Create a Meshes object for the teapot. Here we have only one mesh in the batch.
    R, T = look_at_view_transform(dist=batch_tensor(dist.T, dim=1, squeeze=True), elev=batch_tensor(
        elev.T, dim=1, squeeze=True), azim=batch_tensor(azim.T, dim=1, squeeze=True))
    exhastion = 0
    while not check_valid_rotation_matrix(R, ORTHOGONAL_THRESHOLD):
        exhastion += 1
        # print("azim: {} elev: {}  , dist:{}".format(batch_tensor(azim.T, dim=1, squeeze=True), batch_tensor(elev.T, dim=1, squeeze=True), batch_tensor(canonical_distance.T, dim=1, squeeze=True)))
        # R  = torch.eye(3, dtype=R.dtype, device=R.device).view(1, 3, 3).expand(R.shape[0], -1, -1)
        R, T = look_at_view_transform(dist=setup["canonical_distance"] * torch.ones((c_batch * setup["nb_views"]), device=lights.device), elev=batch_tensor(elev.T + 90.0 * torch.rand_like(elev.T, device=lights.device),
                                                                                                                                                            dim=1, squeeze=True), azim=batch_tensor(azim.T + 180.0 * torch.rand_like(azim.T, device=lights.device), dim=1, squeeze=True))
        print("PROBLEM is fixed {} ? : ".format(exhastion),
              check_valid_rotation_matrix(R, ORTHOGONAL_THRESHOLD))
        if not check_valid_rotation_matrix(R, ORTHOGONAL_THRESHOLD) and exhastion > EXAHSTION_LIMIT:
            sys.exit("REmedy did not work")

    cameras = OpenGLPerspectiveCameras(device="cuda:{}".format(torch.cuda.current_device()), R=R, T=T)
    camera = OpenGLPerspectiveCameras(device="cuda:{}".format(torch.cuda.current_device()), R=R[None, 0, ...],
                                      T=T[None, 0, ...])

    # camera2 = OpenGLPerspectiveCameras(device=device, R=R[None, 2, ...],T=T[None, 2, ...])
    # print(camera2.get_camera_center())
    raster_settings = RasterizationSettings(
        image_size=setup["image_size"],
        blur_radius=0.0,
        faces_per_pixel=1,
        # bin_size=None, #int
        # max_faces_per_bin=None,  # int
        # perspective_correct=False,
        # clip_barycentric_coords=None, #bool
        cull_backfaces=setup["cull_backfaces"],
    )
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=camera, raster_settings=raster_settings),
        shader=HardPhongShader(blend_params=BlendParams(background_color=background_color
                                                        ), device=lights.device, cameras=camera, lights=lights)
    )
    new_meshes = new_meshes.extend(setup["nb_views"])

    # compute output
    # print("after rendering .. ", rendered_images.shape)

    rendered_images = renderer(new_meshes, cameras=cameras, lights=lights)

    rendered_images = unbatch_tensor(
        rendered_images, batch_size=setup["nb_views"], dim=1, unsqueeze=True).transpose(0, 1)
    # print(rendered_images[:, 100, 100, 0])

    rendered_images = rendered_images[..., 0:3].transpose(2, 4).transpose(3, 4)
    return rendered_images, cameras


def render_points(points, color, azim, elev, dist, setup, background_color=(0.0, 0.0, 0.0), device="cuda:0"):
    c_batch = azim.shape[0]

    point_cloud = Pointclouds(points=points.to(torch.float), features=color *
                              torch.ones_like(points, dtype=torch.float)).cuda()

    # print(len(new_meshes.faces_list()[0]))
    # Create a Meshes object for the teapot. Here we have only one mesh in the batch.
    R, T = look_at_view_transform(dist=batch_tensor(dist.T, dim=1, squeeze=True), elev=batch_tensor(
        elev.T, dim=1, squeeze=True), azim=batch_tensor(azim.T, dim=1, squeeze=True))
    exhastion = 0
    while not check_valid_rotation_matrix(R, ORTHOGONAL_THRESHOLD):
        exhastion += 1
        # print("azim: {} elev: {}  , dist:{}".format(batch_tensor(azim.T, dim=1, squeeze=True), batch_tensor(elev.T, dim=1, squeeze=True), batch_tensor(canonical_distance.T, dim=1, squeeze=True)))
        # R  = torch.eye(3, dtype=R.dtype, device=R.device).view(1, 3, 3).expand(R.shape[0], -1, -1)
        R, T = look_at_view_transform(dist=setup["canonical_distance"] * torch.ones((c_batch * setup["nb_views"])).cuda(), elev=batch_tensor(elev.T + 90.0 * torch.rand_like(elev.T).cuda(),
                                                                                                                                             dim=1, squeeze=True), azim=batch_tensor(azim.T + 180.0 * torch.rand_like(azim.T).cuda(), dim=1, squeeze=True))
        print("PROBLEM is fixed {} ? : ".format(exhastion),
              check_valid_rotation_matrix(R, ORTHOGONAL_THRESHOLD))
        if not check_valid_rotation_matrix(R, ORTHOGONAL_THRESHOLD) and exhastion > EXAHSTION_LIMIT:
            sys.exit("REmedy did not work")

    cameras = OpenGLOrthographicCameras(device="cuda:{}".format(torch.cuda.current_device()), R=R, T=T, znear=0.01)
    raster_settings = PointsRasterizationSettings(
        image_size=setup["image_size"],
        radius=setup["points_radius"],
        points_per_pixel=setup["points_per_pixel"]
    )

    renderer = PointsRenderer(
        rasterizer=PointsRasterizer(
            cameras=cameras, raster_settings=raster_settings),
        compositor=NormWeightedCompositor()
    )
    point_cloud = point_cloud.extend(setup["nb_views"])
    point_cloud.scale_(batch_tensor(1.0/dist.T, dim=1,squeeze=True)[..., None][..., None])

    rendered_images = renderer(point_cloud)
    rendered_images = unbatch_tensor(
        rendered_images, batch_size=setup["nb_views"], dim=1, unsqueeze=True).transpose(0, 1)

    rendered_images = rendered_images[..., 0:3].transpose(2, 4).transpose(3, 4)
    return rendered_images, cameras


def auto_render_meshes(targets, meshes, extra_info, correction_factor, models_bag, setup, device="cuda:0"):
    # inputs = inputs.cuda(device)
    # inputs = Variable(inputs)
    c_batch = len(targets)
    # if the model in test phase use white color
    if setup["object_color"] == "random" and not models_bag["view_selector"].training:
        color = torch_color("white")
    else:
        color = torch_color(setup["object_color"],max_lightness=True, epsilon=EPSILON)
    background_color = torch_color(setup["background_color"], max_lightness=True, epsilon=EPSILON).cuda()
    shape_features = models_bag["feature_extractor"](
        extra_info, c_batch_size=c_batch).cuda()
    azim, elev, dist = models_bag["view_selector"](
        shape_features, batch_size=c_batch)

    # lights = PointLights(
    #     device=None, location=((0, 0, 0),))
    if not setup["pc_rendering"]:
        lights = DirectionalLights(
            device=background_color.device, direction=models_bag["view_selector"].light_direction(azim, elev, dist, correction_factor))

        rendered_images, cameras = render_meshes(
            meshes=meshes, color=color, azim=azim, elev=elev, dist=dist, lights=lights, setup=setup, device=None, background_color=background_color)
    else:
        rendered_images, cameras = render_points(
            points=extra_info, color=color, azim=azim, elev=elev, dist=dist, setup=setup, device=None, background_color=background_color)
    #### To perform dropout on the views 
    rendered_images = nn.functional.dropout2d(
        rendered_images, p=setup["view_reg"], training=models_bag["view_selector"].training)
    
    if setup["augment_training"] and models_bag["view_selector"].training:
        rendered_images = super_batched_op(
            1, applied_transforms, rendered_images, crop_ratio=setup["crop_ratio"])
    return rendered_images, cameras, azim, elev, dist


def auto_render_meshes_custom_views(targets, meshes, extra_info, correction_factor, models_bag, setup, device="cuda:0"):
    c_batch = len(targets)
    if setup["object_color"] == "random" and not models_bag["view_selector"].training:
        color = torch_color("white")
    else:
        color = torch_color(setup["object_color"],max_lightness=True, epsilon=EPSILON)

    shape_features = models_bag["feature_extractor"](
        extra_info, c_batch_size=c_batch).cuda()
    azim, elev, dist = models_bag["view_selector"](
        shape_features, batch_size=c_batch)

    for i, target in enumerate(targets.numpy().tolist()):
        azim[i] = torch.from_numpy(np.array(models_bag["azim_dict"][target]))
        elev[i] = torch.from_numpy(np.array(models_bag["elev_dict"][target]))

    # lights = PointLights(
    #     device=device, location=((0, 0, 0),))
    if not setup["pc_rendering"]:
        lights = DirectionalLights(
            targets.device, direction=models_bag["view_selector"].light_direction(azim, elev, dist, correction_factor))

        rendered_images, cameras = render_meshes(
            meshes=meshes, color=color, azim=azim, elev=elev, dist=dist, lights=lights, setup=setup, device=None)
    else:
        rendered_images, cameras = render_points(
            points=extra_info, azim=azim, elev=elev, dist=dist, setup=setup, device=None)
    rendered_images = nn.functional.dropout2d(
        rendered_images, p=setup["view_reg"], training=models_bag["view_selector"].training)

    return rendered_images, cameras, azim, elev, dist


def auto_render_and_save_images_and_cameras(targets, meshes, extra_info, correction_factor, images_path, cameras_path, models_bag, setup, device="cuda:0"):
    # inputs = np.stack(inputs, axis=0)
    # inputs = torch.from_numpy(inputs)
    with torch.no_grad():
        if not setup["return_points_saved"] and not setup["return_points_sampled"]:
            extra_info = torch.from_numpy(extra_info)
        targets = torch.tensor(targets)[None]
        correction_factor = torch.tensor(correction_factor)
        rendered_images, cameras, _, _, _ = auto_render_meshes(
            targets, [meshes], extra_info[None, ...], correction_factor[None, ...], models_bag, setup, device=None)
    # print("before saving .. ",rendered_images.shape)
    save_grid(image_batch=rendered_images[0, ...],
              save_path=images_path, nrow=setup["nb_views"])
    save_cameras(cameras, save_path=cameras_path, scale=0.22, dpi=200)


def auto_render_and_analyze_images(targets, meshes, extra_info, correction_factor, images_path, cameras_path, models_bag, setup, device="cuda:0"):
    # inputs = np.stack(inputs, axis=0)
    # inputs = torch.from_numpy(inputs)
    with torch.no_grad():
        if not setup["return_points_saved"] and not setup["return_points_sampled"]:
            extra_info = torch.from_numpy(extra_info)
        targets = torch.tensor(targets)
        correction_factor = torch.tensor(correction_factor)
        rendered_images, _, _, _, _ = auto_render_meshes(
            targets, [meshes], extra_info[None, ...], correction_factor[None, ...], models_bag, setup, device=None)
    # print("before saving .. ",rendered_images.shape)
    save_grid(image_batch=rendered_images[0, ...],
              save_path=images_path, nrow=setup["nb_views"])
    mask = rendered_images != 1
    img_avg = (rendered_images*mask).sum(dim=(0, 1, 2, 3, 4)) / \
        mask.sum(dim=(0, 1, 2, 3, 4))
    # print("Original  ", rendered_images.mean(dim=(0,1,2,3,4)), "\n IMG avg : ", img_avg)
    # print(img_avg.cpu().numpy())

    return float(img_avg.cpu().numpy())
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


class CanonicalViewSelector(nn.Module):
    def __init__(self, nb_views=12, canonical_elevation=35.0, canonical_distance=2.2, shape_features_size=512, transform_distance=False,input_view_noise=0.0):
        super().__init__()
        self.nb_views = nb_views
        self.transform_distance = transform_distance
        self.canonical_distance = canonical_distance
        self.input_view_noise = input_view_noise
        views_dist = torch.ones(
            (self.nb_views), dtype=torch.float, requires_grad=False) * canonical_distance
        views_azim = torch.linspace(-180, 180, self.nb_views+1)[:-1] - 90.0
        views_elev = torch.ones_like(views_azim, dtype=torch.float, requires_grad=False)*canonical_elevation
        self.register_buffer('views_azim', views_azim)
        self.register_buffer('views_elev', views_elev)
        self.register_buffer('views_dist', views_dist)

    def forward(self, shape_features=None, batch_size=1):
        c_views_azim = self.views_azim.expand(batch_size, self.nb_views)
        c_views_elev = self.views_elev.expand(batch_size, self.nb_views)
        c_views_dist = self.views_dist.expand(batch_size, self.nb_views)
        c_views_dist = c_views_dist + float(self.transform_distance) * 1.0 * c_views_dist * (
            torch.rand((batch_size, self.nb_views), device=c_views_dist.device) - 0.5)
        if self.input_view_noise > 0.0 and self.training:
            c_views_azim = c_views_azim + torch.normal(0.0, 180.0 * self.input_view_noise,c_views_azim.size(), device=c_views_azim.device)
            c_views_elev = c_views_elev + torch.normal(0.0, 90.0 * self.input_view_noise,c_views_elev.size(), device=c_views_elev.device)
            c_views_dist = c_views_dist + torch.normal(0.0, self.canonical_distance * self.input_view_noise,c_views_dist.size(), device=c_views_dist.device)
        return c_views_azim, c_views_elev, c_views_dist


class SphericalViewSelector(nn.Module):
    def __init__(self, nb_views=12, canonical_elevation=35.0, canonical_distance=2.2, shape_features_size=512, transform_distance=False, input_view_noise=0.0):
        super().__init__()
        self.nb_views = nb_views
        self.transform_distance = transform_distance
        self.canonical_distance = canonical_distance
        self.input_view_noise = input_view_noise
        views_dist = torch.ones(
            (self.nb_views), dtype=torch.float, requires_grad=False) * canonical_distance
        views_azim, views_elev = unit_spherical_grid(self.nb_views)
        views_azim, views_elev = torch.from_numpy(views_azim).to(
            torch.float), torch.from_numpy(views_elev).to(torch.float)
        self.register_buffer('views_azim', views_azim)
        self.register_buffer('views_elev', views_elev)
        self.register_buffer('views_dist', views_dist)

    def forward(self, shape_features=None, batch_size=1):
        c_views_azim = self.views_azim.expand(batch_size, self.nb_views)
        c_views_elev = self.views_elev.expand(batch_size, self.nb_views)
        c_views_dist = self.views_dist.expand(batch_size, self.nb_views)
        c_views_dist = c_views_dist + float(self.transform_distance) * 1.0 * c_views_dist * (
            torch.rand((batch_size, self.nb_views), device=c_views_dist.device) - 0.5)
        if self.input_view_noise > 0.0 and self.training:
            c_views_azim = c_views_azim + \
                torch.normal(0.0, 180.0 * self.input_view_noise,
                             c_views_azim.size(), device=c_views_azim.device)
            c_views_elev = c_views_elev + \
                torch.normal(0.0, 90.0 * self.input_view_noise,
                             c_views_elev.size(), device=c_views_elev.device)
            c_views_dist = c_views_dist + \
                torch.normal(0.0, self.canonical_distance * self.input_view_noise,
                             c_views_dist.size(), device=c_views_dist.device)
        return c_views_azim, c_views_elev, c_views_dist

class RandomViewSelector(nn.Module):
    def __init__(self, nb_views=12, canonical_elevation=35.0, canonical_distance=2.2, shape_features_size=512, transform_distance=False):
        super().__init__()
        self.nb_views = nb_views
        self.transform_distance = transform_distance
        self.canonical_distance = canonical_distance
        views_dist = torch.ones((self.nb_views), dtype=torch.float, requires_grad=False) * canonical_distance
        views_elev = torch.zeros((self.nb_views), dtype=torch.float, requires_grad=False)
        views_azim = torch.zeros((self.nb_views), dtype=torch.float, requires_grad=False)
        self.register_buffer('views_azim', views_azim)
        self.register_buffer('views_elev', views_elev)
        self.register_buffer('views_dist', views_dist)

    def forward(self, shape_features=None, batch_size=1):
        c_views_azim = self.views_azim.expand(batch_size, self.nb_views)
        c_views_elev = self.views_elev.expand(batch_size, self.nb_views)
        c_views_dist = self.views_dist.expand(batch_size, self.nb_views)
        c_views_azim = c_views_azim +  torch.rand((batch_size, self.nb_views), device=c_views_azim.device) * 360.0 - 180.0
        c_views_elev = c_views_elev + torch.rand((batch_size, self.nb_views), device=c_views_elev.device) * 180.0 - 90.0
        c_views_dist = c_views_dist +  float(self.transform_distance) * 1.0 * c_views_dist * (torch.rand((batch_size, self.nb_views), device=c_views_dist.device) - 0.499)
        return c_views_azim, c_views_elev, c_views_dist


class MVTDirectViewSelector(nn.Module):
    def __init__(self, nb_views=12, canonical_elevation=35.0, canonical_distance=2.2, shape_features_size=512, transform_distance=False):
        super().__init__()
        self.nb_views = nb_views
        self.transform_distance = transform_distance
        self.canonical_distance = canonical_distance
        views_dist = torch.ones((self.nb_views), dtype=torch.float, requires_grad=False) * canonical_distance
        views_azim = torch.zeros((self.nb_views), dtype=torch.float, requires_grad=False)
        views_elev = torch.zeros((self.nb_views), dtype=torch.float, requires_grad=False)
        if self.transform_distance:
            self.view_transformer = Seq(MLP([shape_features_size, shape_features_size, shape_features_size, 5 *
                                             self.nb_views, 3*self.nb_views],dropout=0.5, norm=True), MLP([3*self.nb_views, 3*self.nb_views], act=None, dropout=0, norm=False), nn.Tanh())
        else:
            self.view_transformer = Seq(MLP([shape_features_size, shape_features_size, shape_features_size, 5 *
                                             self.nb_views, 2*self.nb_views], dropout=0.5, norm=True), MLP([2*self.nb_views, 2*self.nb_views], act=None, dropout=0, norm=False), nn.Tanh())

        self.register_buffer('views_azim', views_azim)
        self.register_buffer('views_elev', views_elev)
        self.register_buffer('views_dist', views_dist)

    def forward(self, shape_features=None, batch_size=1):
        c_views_azim = self.views_azim.expand(batch_size, self.nb_views)
        c_views_elev = self.views_elev.expand(batch_size, self.nb_views)
        c_views_dist = self.views_dist.expand(batch_size, self.nb_views)
        if not self.transform_distance:
            adjutment_vector = self.view_transformer(shape_features)
            adjutment_vector = torch.chunk(adjutment_vector, 2, dim=1)
            return c_views_azim + adjutment_vector[0] * 180.0,  c_views_elev + adjutment_vector[1] * 89.9, c_views_dist
        else:
            adjutment_vector = self.view_transformer(shape_features)
            adjutment_vector = torch.chunk(adjutment_vector, 3, dim=1)
            return c_views_azim + adjutment_vector[0] * 180.0,  c_views_elev + adjutment_vector[1] * 89.9, c_views_dist + adjutment_vector[2] * c_views_dist +0.1


class MVTOffsetViewSelector(nn.Module):
    def __init__(self, nb_views=12, canonical_elevation=35.0, canonical_distance=2.2, shape_features_size=512, transform_distance=False, input_view_noise=0.0):
        super().__init__()
        self.nb_views = nb_views
        self.transform_distance = transform_distance
        self.canonical_distance = canonical_distance
        self.input_view_noise = input_view_noise
        views_dist = torch.ones(
            (self.nb_views), dtype=torch.float, requires_grad=False) * canonical_distance
        views_azim = torch.linspace(-180, 180, self.nb_views+1)[:-1]
        views_elev = torch.ones_like(
            views_azim, dtype=torch.float, requires_grad=False)*canonical_elevation
        if self.transform_distance:
            self.view_transformer = Seq(MLP([shape_features_size+3*self.nb_views, shape_features_size, shape_features_size, 5 *
                                             self.nb_views, 3*self.nb_views], dropout=0.5, norm=True), MLP([3*self.nb_views, 3*self.nb_views], act=None, dropout=0, norm=False), nn.Tanh())
        else:
            self.view_transformer = Seq(MLP([shape_features_size+2*self.nb_views, shape_features_size, shape_features_size, 5 *
                                             self.nb_views, 2*self.nb_views], dropout=0.5, norm=True), MLP([2*self.nb_views, 2*self.nb_views], act=None, dropout=0, norm=False), nn.Tanh())

        self.register_buffer('views_azim', views_azim)
        self.register_buffer('views_elev', views_elev)
        self.register_buffer('views_dist', views_dist)

    def forward(self, shape_features=None, batch_size=1):
        c_views_azim = self.views_azim.expand(batch_size, self.nb_views)
        c_views_elev = self.views_elev.expand(batch_size, self.nb_views)
        c_views_dist = self.views_dist.expand(batch_size, self.nb_views)
        if self.input_view_noise > 0.0 and self.training:
            c_views_azim = c_views_azim + torch.normal(0.0, 180.0 * self.input_view_noise,c_views_azim.size(), device=c_views_azim.device)
            c_views_elev = c_views_elev + torch.normal(0.0, 90.0 * self.input_view_noise,c_views_elev.size(), device=c_views_elev.device)
            c_views_dist = c_views_dist + torch.normal(0.0, self.canonical_distance * self.input_view_noise,
                             c_views_dist.size(), device=c_views_dist.device)

        if not self.transform_distance:
            adjutment_vector = self.view_transformer(
                torch.cat([shape_features, c_views_azim, c_views_elev], dim=1))
            adjutment_vector = torch.chunk(adjutment_vector, 2, dim=1)
            return c_views_azim + adjutment_vector[0] * 180.0/self.nb_views,  c_views_elev + adjutment_vector[1] * 90.0, c_views_dist
        else:
            adjutment_vector = self.view_transformer(
                torch.cat([shape_features, c_views_azim, c_views_elev, c_views_dist], dim=1))
            adjutment_vector = torch.chunk(adjutment_vector, 3, dim=1)
            return c_views_azim + adjutment_vector[0] * 180.0/self.nb_views,  c_views_elev + adjutment_vector[1] * 90.0, c_views_dist + adjutment_vector[2] * self.canonical_distance + 0.1


class MVTSphericalViewSelector(nn.Module):
    def __init__(self, nb_views=12, canonical_elevation=35.0, canonical_distance=2.2, shape_features_size=512, transform_distance=False, input_view_noise=0.0):
        super().__init__()
        self.nb_views = nb_views
        self.transform_distance = transform_distance
        self.canonical_distance = canonical_distance
        self.input_view_noise = input_view_noise
        views_dist = torch.ones(
            (self.nb_views), dtype=torch.float, requires_grad=False) * canonical_distance
        views_azim, views_elev = unit_spherical_grid(self.nb_views)
        views_azim, views_elev = torch.from_numpy(views_azim).to(
            torch.float), torch.from_numpy(views_elev).to(torch.float)
        if self.transform_distance:
            self.view_transformer = Seq(MLP([shape_features_size+3*self.nb_views, shape_features_size, shape_features_size, 5 *
                                             self.nb_views, 3*self.nb_views], dropout=0.5, norm=True), MLP([3*self.nb_views, 3*self.nb_views], act=None, dropout=0, norm=False), nn.Tanh())
        else:
            self.view_transformer = Seq(MLP([shape_features_size+2*self.nb_views, shape_features_size, shape_features_size, 5 *
                                             self.nb_views, 2*self.nb_views], dropout=0.5, norm=True), MLP([2*self.nb_views, 2*self.nb_views], act=None, dropout=0, norm=False), nn.Tanh())

        self.register_buffer('views_azim', views_azim)
        self.register_buffer('views_elev', views_elev)
        self.register_buffer('views_dist', views_dist)

    def forward(self, shape_features=None, batch_size=1):
        c_views_azim = self.views_azim.expand(batch_size, self.nb_views)
        c_views_elev = self.views_elev.expand(batch_size, self.nb_views)
        c_views_dist = self.views_dist.expand(batch_size, self.nb_views)
        c_views_dist = c_views_dist + float(self.transform_distance) * 1.0 * c_views_dist * (
            torch.rand((batch_size, self.nb_views), device=c_views_dist.device) - 0.5)
        if self.input_view_noise > 0.0 and self.training:
            c_views_azim = c_views_azim + \
                torch.normal(0.0, 180.0 * self.input_view_noise,
                             c_views_azim.size(), device=c_views_azim.device)
            c_views_elev = c_views_elev + \
                torch.normal(0.0, 90.0 * self.input_view_noise,
                             c_views_elev.size(), device=c_views_elev.device)
            c_views_dist = c_views_dist + \
                torch.normal(0.0, self.canonical_distance * self.input_view_noise,
                             c_views_dist.size(), device=c_views_dist.device)
        if not self.transform_distance:
            adjutment_vector = self.view_transformer(
                torch.cat([shape_features, c_views_azim, c_views_elev], dim=1))
            adjutment_vector = torch.chunk(adjutment_vector, 2, dim=1)
            return c_views_azim + adjutment_vector[0] * 180.0/self.nb_views,  c_views_elev + adjutment_vector[1] * 90.0, c_views_dist
        else:
            adjutment_vector = self.view_transformer(
                torch.cat([shape_features, c_views_azim, c_views_elev, c_views_dist], dim=1))
            adjutment_vector = torch.chunk(adjutment_vector, 3, dim=1)
            return c_views_azim + adjutment_vector[0] * 180.0/self.nb_views,  c_views_elev + adjutment_vector[1] * 90.0, c_views_dist + adjutment_vector[2] * self.canonical_distance + 0.1


class MVTRandomViewSelector(nn.Module):
    def __init__(self, nb_views=12, canonical_elevation=35.0, canonical_distance=2.2, shape_features_size=512, transform_distance=False, input_view_noise=0.0):
        super().__init__()
        self.nb_views = nb_views
        self.transform_distance = transform_distance
        self.canonical_distance = canonical_distance
        views_dist = torch.ones(
            (self.nb_views), dtype=torch.float, requires_grad=False) * canonical_distance
        views_elev = torch.zeros(
            (self.nb_views), dtype=torch.float, requires_grad=False)
        views_azim = torch.zeros(
            (self.nb_views), dtype=torch.float, requires_grad=False)
        if self.transform_distance:
            self.view_transformer = Seq(MLP([shape_features_size+3*self.nb_views, shape_features_size, shape_features_size, 5 *
                                             self.nb_views, 3*self.nb_views], dropout=0.5, norm=True), MLP([3*self.nb_views, 3*self.nb_views], act=None, dropout=0, norm=False), nn.Tanh())
        else:
            self.view_transformer = Seq(MLP([shape_features_size+2*self.nb_views, shape_features_size, shape_features_size, 5 *
                                             self.nb_views, 2*self.nb_views], dropout=0.5, norm=True), MLP([2*self.nb_views, 2*self.nb_views], act=None, dropout=0, norm=False), nn.Tanh())

        self.register_buffer('views_azim', views_azim)
        self.register_buffer('views_elev', views_elev)
        self.register_buffer('views_dist', views_dist)

    def forward(self, shape_features=None, batch_size=1):
        c_views_azim = self.views_azim.expand(batch_size, self.nb_views)
        c_views_elev = self.views_elev.expand(batch_size, self.nb_views)
        c_views_dist = self.views_dist.expand(batch_size, self.nb_views)
        c_views_azim = c_views_azim + torch.rand((batch_size, self.nb_views),device=c_views_azim.device) * 360.0 - 180.0
        c_views_elev = c_views_elev + torch.rand((batch_size, self.nb_views),device=c_views_elev.device) * 180.0 - 90.0
        c_views_dist = c_views_dist + float(self.transform_distance) * 1.0 * c_views_dist * (torch.rand((batch_size, self.nb_views), device=c_views_dist.device) - 0.499)
        if not self.transform_distance:
            adjutment_vector = self.view_transformer(
                torch.cat([shape_features, c_views_azim, c_views_elev], dim=1))
            adjutment_vector = torch.chunk(adjutment_vector, 2, dim=1)
            return c_views_azim + adjutment_vector[0] * 180.0/self.nb_views,  c_views_elev + adjutment_vector[1] * 90.0, c_views_dist
        else:
            adjutment_vector = self.view_transformer(
                torch.cat([shape_features, c_views_azim, c_views_elev, c_views_dist], dim=1))
            adjutment_vector = torch.chunk(adjutment_vector, 3, dim=1)
            return c_views_azim + adjutment_vector[0] * 180.0/self.nb_views,  c_views_elev + adjutment_vector[1] * 90.0, c_views_dist + adjutment_vector[2] * self.canonical_distance + 0.1

class ViewSelector(nn.Module):
    def __init__(self, nb_views=12, selection_type="canonical", canonical_elevation=30.0, canonical_distance=2.2, shape_features_size=512, transform_distance=False, input_view_noise=0.0, light_direction="fixed"):
        super().__init__()
        self.selection_type = selection_type
        self.nb_views = nb_views
        if self.selection_type == "canonical" or self.selection_type == "custom" or (self.selection_type == "spherical" and self.nb_views == 4):
            self.chosen_view_selector = CanonicalViewSelector(nb_views=nb_views, canonical_elevation=canonical_elevation,
                                                              canonical_distance=canonical_distance, shape_features_size=shape_features_size, transform_distance=transform_distance, input_view_noise=input_view_noise)
        elif self.selection_type == "spherical":
            self.chosen_view_selector = SphericalViewSelector(nb_views=nb_views, canonical_elevation=canonical_elevation,
                               canonical_distance=canonical_distance, shape_features_size=shape_features_size, transform_distance=transform_distance, input_view_noise=input_view_noise)
        elif self.selection_type == "random":
            self.chosen_view_selector = RandomViewSelector(nb_views=nb_views, canonical_elevation=canonical_elevation,canonical_distance=canonical_distance, shape_features_size=shape_features_size, transform_distance=transform_distance)
        elif self.selection_type == "learned_offset" or (self.selection_type == "learned_spherical" and self.nb_views == 4):
            self.chosen_view_selector = MVTOffsetViewSelector(nb_views=nb_views, canonical_elevation=canonical_elevation,
                                                              canonical_distance=canonical_distance, shape_features_size=shape_features_size, transform_distance=transform_distance, input_view_noise=input_view_noise)
        elif self.selection_type == "learned_direct":
            self.chosen_view_selector = MVTDirectViewSelector(nb_views=nb_views, canonical_elevation=canonical_elevation,
                                                              canonical_distance=canonical_distance, shape_features_size=shape_features_size, transform_distance=transform_distance)
        elif self.selection_type  == "learned_spherical":
            self.chosen_view_selector = MVTSphericalViewSelector(nb_views=nb_views, canonical_elevation=canonical_elevation,
                                                                  canonical_distance=canonical_distance, shape_features_size=shape_features_size, transform_distance=transform_distance)
        elif self.selection_type == "learned_random":
            self.chosen_view_selector = MVTRandomViewSelector(nb_views=nb_views, canonical_elevation=canonical_elevation,
                                                              canonical_distance=canonical_distance, shape_features_size=shape_features_size, transform_distance=transform_distance, input_view_noise=input_view_noise)

        self.light_direction_type = light_direction

    def forward(self, shape_features=None, batch_size=1):
        return  self.chosen_view_selector(shape_features=shape_features, batch_size=batch_size)

    def light_direction(self, azim, elev, dist,correction_factor):
        c_bacth_size = azim.shape[0]
        if self.light_direction_type == "fixed":
            return ((0, 1.0, 0),)
        elif self.light_direction_type == "random" and self.training:
            return (tuple(1.0 - 2 * np.random.rand(3)),)
        else:# self.light_direction_type == "relative":
         relative_view = Variable(camera_position_from_spherical_angles(distance=batch_tensor(dist.T, dim=1, squeeze=True), elevation=batch_tensor(elev.T, dim=1, squeeze=True), azimuth=batch_tensor(azim.T, dim=1, squeeze=True))).to(torch.float)
         return correction_factor.repeat_interleave((self.nb_views))[..., None].repeat(1, 3).to(torch.float) * relative_view
        #  

 

def load_point_ckpt(model,  setup,  ckpt_dir='./checkpoint',verbose=True):
    # ------------------ load ckpt
    filename = '{}/{}_model.pth'.format(ckpt_dir, setup["shape_extractor"])
    if not os.path.exists(filename):
        print("No such checkpoint file as:  {}".format(filename))
        return None
    state = torch.load(filename)
    state['state_dict'] = {k: v.cuda() for k, v in state['state_dict'].items()}
    model.load_state_dict(state['state_dict'])
    # optimizer.load_state_dict(state['optimizer_state_dict'])
    # scheduler.load_state_dict(state['scheduler_state_dict'])
    if verbose:
        print('Succeefullly loaded model from {}'.format(filename))





def test_point_network(model,criterion, data_loader,setup,device):
    total = 0.0
    correct = 0.0
    total_loss = 0.0
    n = 0
    from tqdm import tqdm
    for i, (targets,_, points,_) in enumerate(tqdm(data_loader)):
        with torch.no_grad():
            points = points.transpose(1, 2).cuda()
            targets = targets.cuda(device)
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


class FeatureExtracter(nn.Module):
    def __init__(self,  setup):
        super().__init__()
        self.features_size = setup["features_size"]
        self.features_type = setup["features_type"]
        if setup["selection_type"] == "canonical" or setup["selection_type"] == "random" or setup["selection_type"] == "spherical" or setup["selection_type"] == "custom":
            self.features_origin = "zeros"
        elif setup["return_extracted_features"]:
            self.features_origin = "pre_extracted"
        elif setup["is_learning_points"]:
            self.features_origin = "points_features"
            if setup["shape_extractor"] == "PointNet":
                self.fe_model = PointNet(40, alignment=True)
            elif setup["shape_extractor"] == "DGCNN":
                self.fe_model = SimpleDGCNN(40)
            if not setup["screatch_feature_extractor"] : 
                print(setup["shape_extractor"])
                load_point_ckpt(self.fe_model,  setup,  ckpt_dir='./checkpoint')
            self.features_order = {"logits": 0,"post_max": 1, "transform_matrix": 2}

    def forward(self, extra_info=None, c_batch_size=1):
        if self.features_origin == "zeros":
            return torch.zeros((c_batch_size, self.features_size))
        elif self.features_origin == "pre_extracted":
            extra_info = Variable(extra_info)
            return extra_info.view(c_batch_size, self.features_size)
        elif self.features_origin == "points_features":
            extra_info = extra_info.transpose(1, 2).to(next(self.fe_model.parameters()).device)
            features = self.fe_model(extra_info)
            if self.features_type == "logits_trans":
                return torch.cat((features[0].view(c_batch_size, -1), features[2].view(c_batch_size, -1)), 1)
            elif self.features_type == "post_max_trans":
                return torch.cat((features[1].view(c_batch_size, -1), features[2].view(c_batch_size, -1)), 1)
            else:
                return features[self.features_order[self.features_type]].view(c_batch_size, -1)


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
        models_bag["view_selector"].load_state_dict(
            checkpoint['view_selector'])
        models_bag["vs_optimizer"].load_state_dict(checkpoint['vs_optimizer'])
    if setup["is_learning_points"]:
        models_bag["feature_extractor"].load_state_dict(
            checkpoint['feature_extractor'])
        models_bag["fe_optimizer"].load_state_dict(checkpoint['fe_optimizer'])
    if "late_fusion_mode" in setup and setup["late_fusion_mode"]:
        models_bag["classifier"].load_state_dict(checkpoint['classifier'])
        models_bag["cls_optimizer"].load_state_dict(checkpoint['cls_optimizer'])
        models_bag["point_network"].load_state_dict(
            checkpoint['point_network'])
        models_bag["fe_optimizer"].load_state_dict(checkpoint['fe_optimizer'])

    models_bag["optimizer"].load_state_dict(checkpoint['optimizer'])


def load_mvtn(setup, models_bag, weights_file):
    # Load checkpoint.
    print('\n==> Loading checkpoint..')
    assert os.path.isfile(weights_file
                          ), 'Error: no checkpoint file found!'
    checkpoint = torch.load(weights_file)
    models_bag["view_selector"].load_state_dict(checkpoint['view_selector'])
    models_bag["feature_extractor"].load_state_dict(
        checkpoint['feature_extractor'])


def load_checkpoint_robustness(setup, models_bag, weights_file):
    # Load checkpoint.
    print('\n==> Loading checkpoint..')
    assert os.path.isfile(weights_file
                          ), 'Error: no checkpoint file found!'

    checkpoint = torch.load(weights_file)
    models_bag["mvnetwork"].load_state_dict(checkpoint['state_dict'])
    if setup["is_learning_views"]:
        models_bag["view_selector"].load_state_dict(
            checkpoint['view_selector'])
    if setup["is_learning_points"]:
        models_bag["feature_extractor"].load_state_dict(
            checkpoint['feature_extractor'])





class ViewMaxAgregate(nn.Module):
    def __init__(self,  model):
        super().__init__()
        self.model = model
    def forward(self, mvimages):
        B,M,C,H,W = mvimages.shape
        pooled_view = torch.max(unbatch_tensor(self.model(batch_tensor(
            mvimages, dim=1, squeeze=True)), B, dim=1, unsqueeze=True), dim=1)[0]
        return pooled_view.squeeze()


class ViewAvgAgregate(nn.Module):
    def __init__(self,  model):
        super().__init__()
        self.model = model
    def forward(self, mvimages):
        B, M, C, H, W = mvimages.shape
        pooled_view = torch.mean(unbatch_tensor(self.model(batch_tensor(mvimages, dim=1, squeeze=True)), B, dim=1, unsqueeze=True), dim=1)
        return pooled_view.squeeze()


class UnrolledDictModel(nn.Module):
    "a helper class that unroll pytorch models that return dictionaries instead of tensors"

    def __init__(self,  model, keyword="out"):
        super().__init__()
        self.model = model
        self.keyword = keyword
    def forward(self, x):
        return self.model(x)[self.keyword]
class MVAgregate(nn.Module):
    def __init__(self,  model, agr_type="max", feat_dim=512, num_classes=1000):
        super().__init__()
        self.agr_type = agr_type
        self.fc = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, num_classes)
        )
        if self.agr_type == "max":
            self.aggregation_model = ViewMaxAgregate(model=model)
        elif self.agr_type == "mean":
            self.aggregation_model = ViewAvgAgregate(model=model)


    def forward(self, mvimages):
        pooled_view = self.aggregation_model(mvimages)
        predictions = self.fc(pooled_view)
        return predictions, pooled_view



class MyPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None):
        super().__init__()
        import timm.models.layers as tlayers

        img_size = tlayers.to_2tuple(img_size)
        patch_size = tlayers.to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.patch_grid = (img_size[0] // patch_size[0],
                           img_size[1] // patch_size[1])
        self.num_patches = self.patch_grid[0] * self.patch_grid[1]

        self.proj = nn.Conv2d(in_chans, embed_dim,
                              kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x

class FullCrossViewAttention(nn.Module):
    def __init__(self,model,  patch_size=16,num_views=1,feat_dim=512, num_classes=1000):
        super().__init__()
        self.model = model 
        self.model.pos_embed = nn.Parameter(torch.cat(
            (self.model.pos_embed[:, 0, :].unsqueeze(1), self.model.pos_embed[:, 1::, :].repeat((1, num_views, 1))), dim=1))
        # self.model.pos_embed.retain_grad()
        self.combine_views = Rearrange('b N c (h p1) (w p2) -> b c (h p1 N) (w p2)',p1=patch_size, p2=patch_size, N=num_views)
        self.fc = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, num_classes)
        )

    def forward(self, mvimages):
        mvimages = self.combine_views(mvimages)
        feats = self.model(mvimages)
        return self.fc(feats), feats

class WindowCrossViewAttention(nn.Module):
    def __init__(self,model,  patch_size=16,num_views=1,num_windows=1,feat_dim=512, num_classes=1000,agr_type="max"):
        super().__init__()

        assert num_views % num_windows == 0 , "the number of winsows should be devidand of number of views "
        view_per_window = int(num_views/num_windows)
        model.pos_embed = nn.Parameter(torch.cat((model.pos_embed[:, 0, :].unsqueeze(1), model.pos_embed[:, 1::, :].repeat((1, view_per_window, 1))), dim=1))
        self.model = MVAgregate(model, agr_type=agr_type,
                                feat_dim=feat_dim, num_classes=num_classes)
        self.combine_views = Rearrange('b (Win NV) c (h p1) (w p2) -> b Win c (h p1 NV) (w p2)',
                                       p1=patch_size, p2=patch_size, Win=num_windows, NV=view_per_window)

    def forward(self, mvimages):
        mvimages = self.combine_views(mvimages)
        pred , feats = self.model(mvimages)
        return pred , feats
def mvtosv(x): return rearrange(x, 'b m h w -> (b m) h w ')
def mvctosvc(x): return rearrange(x, 'b m c h w -> (b m) c h w ')
def svtomv(x,nb_views=1): return rearrange(x, '(b m) h w -> b m h w',m=nb_views)
def svctomvc(x,nb_views=1): return rearrange(x, '(b m) c h w -> b m c h w',m=nb_views)


class MVPartSegmentation(nn.Module):
    def __init__(self,  model, num_classes,parts_per_class,parallel_head=False):
        super().__init__()
        self.num_classes = num_classes
        self.model = model
        self.multi_shape_heads = nn.ModuleList()
        self.parallel_head = parallel_head
        if parallel_head:
            for cls in range(num_classes):
                self.multi_shape_heads.append(nn.Sequential(torch.nn.Conv2d(21, 2*max(parts_per_class), kernel_size=(1, 1), stride=(1, 1)),
                                    nn.BatchNorm2d(2*max(parts_per_class)),
                                    nn.ReLU(inplace=True),
                    torch.nn.Conv2d(2*max(parts_per_class), max(parts_per_class)+1, kernel_size=(1, 1), stride=(1, 1))
                                    ))
        else:
            self.multi_shape_heads.append(nn.Sequential(torch.nn.Conv2d(21, 21, kernel_size=(1, 1), stride=(1, 1)),
                    nn.BatchNorm2d(21),
                    nn.ReLU(inplace=True),
                    torch.nn.Conv2d(21, max(parts_per_class)+1, kernel_size=(1, 1), stride=(1, 1))
                    ))
    def forward(self, mvimages):
        features = self.model(mvctosvc(mvimages))["out"]
        if self.parallel_head:
            logits_all_shapes = []
            for cls in range(self.num_classes):
                logits_all_shapes.append(self.multi_shape_heads[cls](features)[...,None])
            return torch.cat(logits_all_shapes, dim=4)
        else: 
            return self.multi_shape_heads[0](features)

def extra_IOU_metrics(points_GT, points_predictions, pixels_GT, pixel_mask, points_mask,object_class, parts,):
    """
    a funciton to calculate IOUs  for bacth of point clouds `points_predictions` based on the ground truth `points_GT` and record more metrics as well
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
