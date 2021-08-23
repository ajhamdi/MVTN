from einops import rearrange
import os
import string
import time
from collections import defaultdict
from multiprocessing import Pool
import random

import trimesh
from matplotlib import pyplot as plt
import torch
from pytorch3d.renderer import DirectionalLights, Textures, look_at_view_transform, \
    FoVPerspectiveCameras, RasterizationSettings, MeshRasterizer, HardPhongShader
from pytorch3d.structures import Meshes
from scipy import sparse
from sklearn.neighbors import KDTree
from scipy.special import softmax
from torch import nn
from torchvision.models.segmentation import deeplabv3_resnet50
import numpy as np
from scipy import stats

from ops import *

LIGHT_DIRECTION = (0, 1.0, 0)


def batch_classes2weights(batch_classes, cls_freq):
    cls_freq = {k: v for k, v in cls_freq}
    class_weight = class_freq_to_weight(cls_freq)
    class_weight = [class_weight[x] for x in sorted(class_weight.keys())]
    c_batch_weights = torch.ones_like(
        batch_classes).cuda() * torch.Tensor(class_weight).cuda()[batch_classes.cpu().numpy().tolist()]
    return c_batch_weights

def render_points_parts(points, color, azim, elev, dist, setup, background_color=(0.0, 0.0, 0.0), ):
    c_batch = azim.shape[0]
    point_cloud = Pointclouds(points=points.to(torch.float).cuda() , features=color *
                              torch.ones_like(points, dtype=torch.float)).cuda()
    R, T = look_at_view_transform(dist=batch_tensor(dist.T, dim=1, squeeze=True), elev=batch_tensor(
        elev.T, dim=1, squeeze=True), azim=batch_tensor(azim.T, dim=1, squeeze=True))

    cameras = OpenGLOrthographicCameras(device="cuda:{}".format(torch.cuda.current_device()), R=R, T=T, znear=0.01)
    raster_settings = PointsRasterizationSettings(
        image_size=setup["image_size"],
        radius=setup["points_radius"],
        points_per_pixel=setup["points_per_pixel"]
    )

    rasterizer = PointsRasterizer(cameras=cameras, raster_settings=raster_settings)
    compositor = NormWeightedCompositor(background_color=background_color)

    point_cloud = point_cloud.extend(setup["nb_views"]) 
    point_cloud.scale_(batch_tensor(1.0/dist.T, dim=1, squeeze=True)[..., None][..., None])
    fragments = rasterizer(point_cloud,)

        # Construct weights based on the distance of a point to the true point.
        # However, this could be done differently: e.g. predicted as opposed
        # to a function of the weights.
    r = rasterizer.raster_settings.radius
    dists2 = fragments.dists.permute(0, 3, 1, 2)
    weights = 1 - dists2 / (r * r)
    rendered_images = compositor(fragments.idx.long().permute(0, 3, 1, 2),weights,point_cloud.features_packed().permute(1, 0),)
        # permute so image comes at the end
    rendered_images = rendered_images.permute(0, 2, 3, 1)

    # rendered_images = renderer(point_cloud)
    rendered_images = unbatch_tensor(
        rendered_images, batch_size=setup["nb_views"], dim=1, unsqueeze=True).transpose(0, 1)
    weights = unbatch_tensor(weights, batch_size=setup["nb_views"], dim=1, unsqueeze=True).transpose(0, 1)
    indxs = unbatch_tensor(fragments.idx.long().permute(0, 3, 1, 2), batch_size=setup["nb_views"], dim=1, unsqueeze=True).transpose(0, 1)

    rendered_images = rendered_images[..., 0:3].transpose(2, 4).transpose(3, 4)
    # weights = weights[..., 0:3].transpose(2, 4).transpose(3, 4)
    # indxs = indxs[..., 0:3].transpose(2, 4).transpose(3, 4)


    return rendered_images, indxs , weights  


def auto_render_parts(targets, meshes, points, models_bag, setup, color=[], ):
    c_batch = len(targets)
    # if the model in test phase use white color
    if len(color)==0: 
        if setup["object_color"] == "random" and not models_bag["mvtn"].training:
            color = torch_color("white")
        else:
            color = torch_color(setup["object_color"],
                                max_lightness=True)
    background_color = torch_color(
        setup["background_color"], max_lightness=True).cuda()
    azim, elev, dist = models_bag["mvtn"](points, c_batch_size=c_batch)

    if not setup["pc_rendering"]:
        # lights = DirectionalLights(
        #     device=device, direction=models_bag["mvtn"].light_direction(azim, elev, dist, correction_factor))

        # rendered_images, cameras = render_meshes(
        #     meshes=meshes, color=color, azim=azim, elev=elev, dist=dist, lights=lights, setup=setup, device=device, background_color=background_color)
        raise NotImplementedError("this is still not implemented no mesh part sgemtnation for now ")
    else:
        rendered_images, indxs, weights = render_points_parts(
            points=points, color=color, azim=azim, elev=elev, dist=dist, setup=setup, background_color=background_color)
    #### To perform dropout on the views
    rendered_images = nn.functional.dropout2d(
        rendered_images, p=setup["view_reg"], training=models_bag["mvtn"].training)

    if setup["augment_training"] and models_bag["mvtn"].training:
        rendered_images = super_batched_op(
            1, applied_transforms, rendered_images, crop_ratio=setup["crop_ratio"])
    return rendered_images, indxs, weights, azim, elev, dist

def timing_val(func):
    def wrapper(*arg, **kw):
        t1 = time.time()
        res = func(*arg, **kw)
        t2 = time.time()
        print('{} took {:.4f}'.format(func.__name__, t2 - t1))
        return res
    return wrapper


class MVTViewRenderer(nn.Module):
    """
    A class for rendering a batch of meshes. Use it to compute rendered images and pix_to_face info.
    """

    def __init__(self, device, args):
        super().__init__()

        self.feature_extractor = FeatureExtracter(args).cuda()

        self.view_selector = ViewSelector(
            nb_views=args["nb_views"],
            views_config=args["views_config"],
            canonical_elevation=args["canonical_elevation"],
            canonical_distance=args["canonical_distance"],
            shape_features_size=args["features_size"],
            transform_distance=args["transform_distance"],
            input_view_noise=args["input_view_noise"],
            light_direction=args["light_direction"]).cuda()

        self.device = device
        self.args = args

    def forward(self, meshes):
        """
        meshes: list[Mesh]
        Return:
            rgb_images: (batch_size * nb_view, h, w, 3). RGB values are in the range of [0, 1].
            pix_to_face: (batch_size * nb_view, h, w)
        """
        batch_size = len(meshes)

        # compute point cloud feature
        points_list = []
        for i in range(batch_size):
            trimesh_obj = trimesh.Trimesh(
                vertices=meshes[i].verts_list()[0].numpy(),
                faces=meshes[i].faces_list()[0].numpy(),
                process=False)
            pc = trimesh_obj.sample(2048, False)  # (2048, 3)
            points_list.append(pc)
        points = torch.tensor(np.stack(points_list), dtype=torch.float).to(self.device)  # (b, n, 3)
        pc_features = self.feature_extractor(points, batch_size)  # (b, n, c)

        # compute azim, elev, dist
        batch_azim, batch_elev, batch_dist = self.view_selector(shape_features=pc_features, batch_size=batch_size)

        device_meshes = Meshes(
            verts=[msh.verts_list()[0].to(self.device) for msh in meshes],
            faces=[msh.faces_list()[0].to(self.device) for msh in meshes],
            textures=None)

        max_vert = device_meshes.verts_padded().shape[1]
        device_meshes.textures = Textures(verts_rgb=torch.ones((batch_size, max_vert, 3)).to(self.device))

        # Create new Meshes which contains each input mesh N times
        # [a, b, c] -> [a, a ..., b, b ..., c, c ...]
        device_meshes = device_meshes.extend(self.args['nb_views'])

        R, T = look_at_view_transform(
            dist=self.args["distance_to_object"],
            elev=batch_elev.flatten(),
            azim=batch_azim.flatten())

        cameras = FoVPerspectiveCameras(device=self.device, R=R, T=T)

        raster_settings = RasterizationSettings(
            image_size=self.args["image_size"],
            blur_radius=0.0,
            faces_per_pixel=1,
        )

        rasterizer = MeshRasterizer(
            cameras=cameras,
            raster_settings=raster_settings)

        lights = DirectionalLights(
            device=self.device,
            direction=self.view_selector.light_direction(batch_azim, batch_elev, batch_dist))

        shader = HardPhongShader(
            device=self.device,
            cameras=cameras,
            lights=lights)

        fragments = rasterizer(device_meshes)
        images = shader(fragments, device_meshes)
        pix_to_face = fragments.pix_to_face
        pix_dist = fragments.zbuf

        # only use first 3 channel
        rgb_images = images[:, :, :, :3]

        # squeeze last dim
        pix_to_face = pix_to_face.squeeze(-1)

        return rgb_images, pix_to_face, pix_dist


class MultiViewSegmentationNet(nn.Module):
    def __init__(self, args, num_classes):
        """
        Segmentation sub-network for rendered images.
        Args:
            args: bunch of params.
            num_classes: int.
        """
        super().__init__()

        if args['network'] == 'deeplabv3_resnet50':
            self.segmentation_net = deeplabv3_resnet50(
                pretrained=args['pretrained'],
                num_classes=num_classes)
        else:
            raise RuntimeError('network type not supported.')

    def forward(self, images):
        """
        Compute segmentation result.
        Args:
            images: of shape (N, 3, H, W), where N is the number of images, H and W are expected to be at least
            224 pixels. The images should be in the range of [0, 1].
            Here, N means batch_size * nb_view.
        Returns:
            An OrderedDict with two Tensors that are of the same height and width as the input Tensor, but with
            num_classes classes. output['out'] contains the semantic masks and is of shape (N, num_classes, H, W).
        """
        result = self.segmentation_net(images)
        return result


class MVTSegNet(nn.Module):
    def __init__(self, args, num_class, device):
        """
        Main network for mesh face segmentation.
        Args:
            args: bunch of params.
            num_class: number of class. Make sure it includes the background class.
            device: device type string.
        """
        super().__init__()

        self.renderer = MVTViewRenderer(device, args)
        self.seg_net = MultiViewSegmentationNet(args, num_class)
        self.seg_net.to(device=device)

    def forward(self, meshes):
        """
        Args:
            meshes: list of N meshes. N is batch_size.
        Returns:
            rendered_images: (N * nb_view, h, w, 3)
            rendered_pix_to_face: (N * nb_view, h, w)
            seg_result: (N * nb_view, num_class, h, w), segmentation result.
        """
        # generate images using mesh from different elevation/azimuth.
        rendered_images, rendered_pix_to_face, rendered_pix_dist = self.renderer(meshes)
        rendered_im_depth = torch.cat([rendered_images[:, :, :, :2], rendered_pix_dist], dim=-1)

        # (b, h, w, 3) -> (b, 3, h, w)
        images = rendered_im_depth.transpose(1, 3)

        # segmentation
        seg_result = self.seg_net(images)

        return rendered_images, rendered_pix_to_face, seg_result, rendered_pix_dist


def batched_index_select(x, idx):
    """
    This can be used for neighbors features fetching
    Given a pointcloud x, return its k neighbors features indicated by a tensor idx.
    :param x: torch.Size([batch_size, num_dims, num_vertices, 1])
    :param index: torch.Size([batch_size, num_vertices, k])
    :return: torch.Size([batch_size, num_dims, num_vertices, k])
    """

    batch_size, num_dims, num_vertices = x.shape[:3]
    _ , all_combo , k = idx.shape
    idx_base = torch.arange(
        0, batch_size, device=idx.device).view(-1, 1, 1) * num_vertices
    idx = idx + idx_base
    idx = idx.view(-1)

    x = x.transpose(2, 1).contiguous()
    feature = x.view(batch_size * num_vertices, -1)[idx, :]
    feature = feature.view(batch_size, all_combo, k,
                           num_dims).permute(0, 3, 1, 2)
    return feature

def batched_index_select_parts(x, idx):
    """
    This can be used for neighbors features fetching
    Given a pointcloud x, return its k neighbors features indicated by a tensor idx.
    :param x: torch.Size([batch_size, num_vertices , 1])
    :param index: torch.Size([batch_size, num_views, points_per_pixel,H,W])
    :return: torch.Size([batch_size, _vertices, k])
    """

        
    batch_size, num_view, num_nbrs , H ,W  = idx.shape[:5]
    _, num_dims , num_vertices = x.shape

    idx = rearrange(idx, 'b m p h w -> b (m h w) p')
    x = x[...,None]
    feature = batched_index_select(x, idx)
    feature = rearrange(feature, 'b d (m h w) p -> b m d p h w',
                        m=num_view, h=H, w=W, d=num_dims)
    return feature
def compute_image_segment_label_points(points, batch_points_labels, rendered_pix_to_point,):
    """
    Compute ground truth segmentation labels for rendered images.
    Args:
        args: bunch of arguments.
        rendered_pix_to_point: (B * nb_view, h, w).
        batch_meshes: list[Mesh], with length of B
        device: device string
        batch_points_labels: Tensor with shape (B, max_face_label_length), padded with -1.
    Returns:
        labels_2d: (B * nb_view, h, w).
        pix_to_face_mask: (B * nb_view, h, w). Valid mask for labels_2d.
    """
    # invalid value in pix_to_face is -1, use it to build a mask.
    # (batch_size * nb_view, h, w)
    ### adjustment 
    B, N, _ = points.shape
    pix_to_face_mask = ~rendered_pix_to_point.eq(-1)
    rendered_pix_to_point = rendered_pix_to_point % N +1 
    rendered_pix_to_point[~pix_to_face_mask] = 0
    batch_points_labels = torch.cat((torch.zeros(B)[...,None].to(torch.int32).cuda(), batch_points_labels),dim=1)

    # unpack pix_to_face
    class_map = batched_index_select_parts(batch_points_labels[:, None, ...], rendered_pix_to_point)
    labels_2d = class_map[:,:,0,0,...] # take only max point label for each pixel 

    return labels_2d, pix_to_face_mask


def lift_2D_to_3D(points, predictions_2d, rendered_pix_to_point, views_weights,  parts_range, parts_nb, lifting_method="mode"):
    """
    Unproject the 2d predictions of segmentation labels to 3d.
    Args:
        rendered_pix_to_point: (B * nb_view, h, w).
        device: device string
    Returns:
        labels_3d: (B * nb_points, ).
    """
    B, N, _ = points.shape
    nb_classes = predictions_2d.shape[2]
    pix_to_face_mask = ~rendered_pix_to_point.eq(-1)
    rendered_pix_to_point = rendered_pix_to_point % N 
    rendered_pix_to_point[~pix_to_face_mask] = 0

    labels_3d = torch.zeros((B, N, torch.max(parts_nb).item(),)).to(
        predictions_2d.device)
    labels_count = torch.zeros((B, N, torch.max(parts_nb).item()), dtype=torch.long).to(predictions_2d.device)
    _, predictions_2d_max = torch.max(predictions_2d, dim=2)
    predictions_2d = torch.nn.functional.softmax(predictions_2d,dim=2)
    predictions_2d = views_weights * predictions_2d

    if lifting_method == "mode":
        for batch in range(B):
            for ii, part in enumerate(range(1, 1 + parts_nb[batch].item())):
                points_indices = rendered_pix_to_point[batch][predictions_2d_max[:, :, None, ...].expand(
                    rendered_pix_to_point.size())[batch] == part]

                points_indices, points_counts = torch.unique(
                    points_indices, return_counts=True)
                labels_count[batch, :, ii][points_indices] = points_counts
                labels_3d[batch, :, ii][points_indices] = part
        _, indcs = torch.max(labels_count, dim=-1)
        empty_points = torch.sum(labels_3d,dim=-1) == 0
        labels_3d = indcs + 1
        labels_3d[empty_points] = 0
    elif lifting_method == "mean" or "attention" in lifting_method:
        labels_3d_feats = torch.zeros((B, N, nb_classes,)).to(predictions_2d.device)
        for batch in range(B):
            for ii, part in enumerate(range(1, 1 + parts_nb[batch].item())):
                class_label_mask = predictions_2d_max[:, :, None, ...].expand(rendered_pix_to_point.size())[batch] == part
                points_indices = rendered_pix_to_point[batch][class_label_mask]
                unique_points_indices, points_counts = torch.unique(points_indices, return_counts=True)
                labels_count[batch, :, ii][unique_points_indices] = points_counts
                labels_3d[batch, :, ii][unique_points_indices] = part
                selected_feats = []
                for jj in range(nb_classes):
                    selected_feats.append(predictions_2d[batch, :, jj, :, :][class_label_mask.squeeze()][...,None])
                selected_feats = torch.cat(selected_feats,dim=1)
                # print(class_label_mask.shape, predictions_2d.shape, selected_feats.shape,labels_3d_feats[batch, :, :][points_indices].shape)
                labels_3d_feats[batch, :, :][points_indices] += selected_feats
        empty_points = torch.sum(labels_3d,dim=-1) == 0
        _, labels_3d = torch.max(labels_3d_feats,dim=-1)
        labels_3d[empty_points] = 0
    elif lifting_method == "mlp":
        pass

    # print("############", "empty points:  ",torch.sum(labels_3d == 0,dim=-1))

    return labels_3d


def knn(x, k):
    """
    Given point features x [B, C, N, 1], and number of neighbors k (int)
    Return the idx for the k neighbors of each point. 
    So, the shape of idx: [B, N, k]
    """
    with torch.no_grad():
        x = x.squeeze(-1)
        inner = -2 * torch.matmul(x.transpose(2, 1), x)
        xx = torch.sum(x ** 2, dim=1, keepdim=True)
        inner = -xx - inner - xx.transpose(2, 1)

        idx = inner.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
    return idx


def knnq2ref(q, ref, k):
    """
    Given query point features x [B, C, N, 1] and ref point features x [B, C, M, 1]  and number of neighbors k (int)
    Return the idx for the k neighbors in ref for all query points . 
    So, the shape of idx: [B, N, k]
    """
    B, C, M, _ = ref.shape
    with torch.no_grad():
        q = q.repeat((1, 1, 1, M))
        ref = ref
        dist = torch.norm(a - ref.transpose(2, 3), dim=1, p=2)
        idx = dist.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
    return idx


def post_process_segmentation(point_set, predictions_3d,iterations=1,K_neighbors=1):
    """
    a function to fill empty points in point cloud `point_set` with the labels of their nearest neghbors in `predictions_3d` in an iterative fashion 
    """
    for iter in range(iterations):
        emptry_points = predictions_3d == 0
        nbr_indx = knn(point_set.transpose(1, 2)[..., None], iter*K_neighbors + 2)
        nbr_labels = batched_index_select(predictions_3d[...,None].transpose(1, 2)[..., None], nbr_indx)
        nbr_labels = torch.mode(nbr_labels[:,0,:,1::],dim=-1)[0] # only look at the closest neighbor to fetch its labels   
        predictions_3d[emptry_points] = nbr_labels[emptry_points]
    return predictions_3d 

def compute_metrics(labels_3d, face_labels, label_range):
    """
    Compute some metrics values for faces.
    Args:
        labels_3d: (face_num, ), predicted face labels. -1 means invalid.
        face_labels: (face_num, ), ground-truth face labels.
        label_range: (2, ), start and end label. End label is included.
    Returns:
        face_cov: face coverage.
        face_acc: face accuracy in all faces.
        cov_face_acc: face accuracy only in covered faces.
        IoU: IoU for one mesh.
    """
    face_cov = np.sum(labels_3d != -1) / labels_3d.shape[0]
    face_acc = np.sum(labels_3d == face_labels) / labels_3d.shape[0]
    cov_faces = labels_3d != -1
    cov_face_acc = np.sum(labels_3d[cov_faces] == face_labels[cov_faces]) / np.sum(cov_faces)

    # compute IoU
    IoU_part_sum = 0.0
    for class_idx in range(label_range[0], label_range[1] + 1):
        location_gt = (face_labels == class_idx)
        location_pred = (labels_3d == class_idx)
        I_locations = np.logical_and(location_gt, location_pred)
        U_locations = np.logical_or(location_gt, location_pred)
        I = np.sum(I_locations) + np.finfo(np.float32).eps
        U = np.sum(U_locations) + np.finfo(np.float32).eps
        IoU_part_sum += I / U

    IoU = IoU_part_sum / (label_range[1] - label_range[0] + 1)
    return face_cov, face_acc, cov_face_acc, IoU


def save_batch_rendered_images(rendered_images, save_dir, image_name):
    """
    Helper function to save rendered batch images for debug purpose.
    rendered_images: [B * nb_view, h, w, c]
    """
    assert len(rendered_images.shape) == 5, "params error"
    batch_size, nb_view, c, h, w = rendered_images.shape
    rendered_images = rearrange(rendered_images, 'b m c h w -> (b m) h w c')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    fig, axs = plt.subplots(
        batch_size,
        nb_view,
        gridspec_kw={"wspace": 0.0, "hspace": 0.0},
        figsize=(nb_view*9, 9)
    )

    fig.subplots_adjust(
        left=0.02,
        bottom=0.02,
        right=0.98,
        top=0.98,
    )
    if batch_size * nb_view == 1 :
        axs.imshow(rendered_images[0].cpu())
        axs.set_xticks([])
        axs.set_yticks([])
    else:
        for i, ax in enumerate(axs.ravel()):
            ax.imshow(rendered_images[i].cpu())
            ax.set_xticks([])
            ax.set_yticks([])

    plt.savefig(os.path.join(save_dir, image_name))
    plt.close("all")

def save_batch_rendered_segmentation_images(seg_labels, save_dir, image_name,given_labels=None, plt_cmap="nipy_spectral"):
    """
    Helper function to save rendered batch images with different segmentation labels . if given_labels != None , show only these labels  ,  
    seg_labels: [B * nb_view, h, w]
    """
    assert len(seg_labels.shape) == 4, "params error"
    batch_size, nb_view, h, w = seg_labels.shape
    avail_classes_nbs = torch.unique(seg_labels, sorted=True).cpu().numpy().tolist()

    if given_labels:
        given_labels =[xx+1 for xx in given_labels]
        for lbl in avail_classes_nbs:
            if lbl not in given_labels:
                seg_labels[seg_labels == lbl] = given_labels[0] - 1
    else:
        if len(avail_classes_nbs) > 1:
            min_class = avail_classes_nbs[1]
        else:
            min_class = 1
        seg_labels[seg_labels == 0] = min_class - 1


    seg_labels = rearrange(seg_labels, 'b m h w -> (b m) h w ')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    fig, axs = plt.subplots(
        batch_size,
        nb_view,
        gridspec_kw={"wspace": 0.0, "hspace": 0.0},
        figsize=(nb_view*9, 9)
    )

    fig.subplots_adjust(
        left=0.02,
        bottom=0.02,
        right=0.98,
        top=0.98,
    )
    if batch_size * nb_view == 1:
        axs.imshow(seg_labels[0].cpu(), cmap=plt_cmap)
        axs.set_xticks([])
        axs.set_yticks([])
    else:
        for i, ax in enumerate(axs.ravel()):
            ax.imshow(seg_labels[i].cpu(), cmap=plt_cmap)
            ax.set_xticks([])
            ax.set_yticks([])

    plt.savefig(os.path.join(save_dir, image_name))
    plt.close("all")
