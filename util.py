import matplotlib.pyplot as plt
import mpl_toolkits
import torch
import math
import numpy as np
import random
import os
import pickle
import copy
import pandas as pd
import imageio
import trimesh
from torch import nn
from torch.nn import Sequential as Seq, Linear as Lin, Conv1d
import matplotlib
matplotlib.use('Agg')


import glob
import json
import yaml



def read_json(file_path):
    """
    read config files 
    """
    with open(file_path, "r") as f:
        return json.load(f)

def read_yaml(file_path):
    """
    read config files
    """
    with open(file_path, "r") as f:
        return yaml.safe_load(f)

def simplify_mesh(input_file, simplify_ratio=0.05):
    """
    a function to reduce the poly of meshe `input_file` by some ratio `simplify_ratio`
    Reuturns : the mesh in `input_file` as Trimesh object and the simplified mesh as Trimeh object and saves the simplified mesh with the 
    based on `https://github.com/HusseinBakri/BlenderPythonDecimator`
    """
    project_dir = os.getcwd()
    if input_file[-3::] == "off":
        input_obj_file = input_file.replace(".off", ".obj")
        input_off_file = input_file
    elif input_file[-3::] == "obj":
        input_off_file = input_file.replace(".obj", ".off")
        input_obj_file = input_file
    mymesh = trimesh.load(input_file)
    input_file = input_file[:-4]
    output_obj_file = "{}_SMPLER.obj".format(input_file)
    if not os.path.isfile(input_obj_file):
        _ = mymesh.export(input_obj_file)
    command = "blender -b -P {} -- --ratio {} --inm '{}' --outm '{}'".format(os.path.join(
        project_dir, "blender_simplify.py"), simplify_ratio, input_obj_file, output_obj_file)
    os.system(command)
    reduced_mesh = trimesh.load(output_obj_file)
    return mymesh,  reduced_mesh

def torch_deg2rad(degs):
    return degs * np.pi/180.0

def torch_rad2deg(rads):
    return rads * 180.0/np.pi


def torch_direction_vector(azim, elev, from_degrees=True):
    """
    a torch util fuinction to convert batch elevation and zimuth angles ( in degrees or radians) to a R^3 direction unit vector
    """
    bs = azim.shape[0]

    if from_degrees:
        azim, elev = torch_deg2rad(azim), torch_deg2rad(elev)
    dir_vector = torch.zeros(bs, 3)
    dir_vector[:, 0] = torch.sin(azim) * torch.cos(elev)
    dir_vector[:, 1] = torch.sin(elev)
    dir_vector[:, 2] = torch.cos(azim) * torch.cos(elev)
    return dir_vector

def class_freq_to_weight(class_freqs):
    """
    a function to convert  a dictionary of labels frequency  to dictionary of loss weights per class label that are averaged to 1. This is helpful in designing a weighted loss 
    """
    total = 0
    result_weights = {}
    cls_nbrs = len(class_freqs)
    for k, v in class_freqs.items():
        total += v
    avg = total/float(cls_nbrs)
    for k, v in class_freqs.items():
        result_weights[k] = avg/v
    return result_weights

def batch_points_mIOU(points_GT,points_predictions,points_mask,parts,):
    """
    a funciton to calculate mIOU for bacth of point clouds `points_predictions` based on the ground truth `points_GT`
    """
    bs = points_GT.shape[0]
    cur_shape_iou_tot = torch.zeros(bs, ).cuda()
    cur_shape_iou_cnt = torch.zeros(bs, ).cuda()
    for cl in range(torch.max(parts).item()):
        # -1 to remove the background class laabel
        cur_gt_mask = (points_GT == cl) & points_mask  # -1 to remove the background class laabel
        cur_pred_mask = (points_predictions == cl) & points_mask

        I = (cur_pred_mask & cur_gt_mask).sum(dim=-1)
        U = (cur_pred_mask | cur_gt_mask).sum(dim=-1)

        # part_intersect[cl] += I.sum()
        # part_union[cl] += U.sum()


        cur_shape_iou_tot += I/(U + 1e-7)
        cur_shape_iou_cnt += (U > 0).to(torch.float)

    cur_shape_miou = cur_shape_iou_tot / cur_shape_iou_cnt
    return cur_shape_miou

# https://github.com/pratogab/batch-transforms
def profile_op(max_iter, operation, *args, **kwargs):
        """
        a util function to profile the speed of a python function `operation` that has inputs `*args,**kwargs` . The average time is using `max_iter` iterations 
        """
        from timeit import default_timer as timer
        start = timer()
        for _ in range(max_iter):
            operation(*args, **kwargs)
        end = timer()
        avg_time = (end - start)/float(max_iter)
        return avg_time
class Normalize:
    """Applies the :class:`~torchvision.transforms.Normalize` transform to a batch of images.
    .. note::
        This transform acts out of place by default, i.e., it does not mutate the input tensor.
    Args:
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
        inplace(bool,optional): Bool to make this operation in-place.
        dtype (torch.dtype,optional): The data type of tensors to which the transform will be applied.
        device (torch.device,optional): The device of tensors to which the transform will be applied.
    """

    def __init__(self, mean, std, inplace=False, dtype=torch.float, device='cpu'):
        self.mean = torch.as_tensor(mean, dtype=dtype, device=device)[
            None, :, None, None]
        self.std = torch.as_tensor(std, dtype=dtype, device=device)[
            None, :, None, None]
        self.inplace = inplace

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor of size (N, C, H, W) to be normalized.
        Returns:
            Tensor: Normalized Tensor.
        """
        if not self.inplace:
            tensor = tensor.clone()

        tensor.sub_(self.mean).div_(self.std)
        return tensor


class RandomHorizontalFlip:
    """Applies the :class:`~torchvision.transforms.RandomHorizontalFlip` transform to a batch of images.
    .. note::
        This transform acts out of place by default, i.e., it does not mutate the input tensor.
    Args:
        p (float): probability of an image being flipped.
        inplace(bool,optional): Bool to make this operation in-place.
    """

    def __init__(self, p=0.5, inplace=False):
        self.p = p
        self.inplace = inplace

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor of size (N, C, H, W) to be flipped.
        Returns:
            Tensor: Randomly flipped Tensor.
        """
        if not self.inplace:
            tensor = tensor.clone()

        flipped = torch.rand(tensor.size(0)) < self.p
        tensor[flipped] = torch.flip(tensor[flipped], [3])
        return tensor


class RandomCrop:
    """Applies the :class:`~torchvision.transforms.RandomCrop` transform to a batch of images.
    Args:
        size (int): Desired output size of the crop.
        padding (int, optional): Optional padding on each border of the image. 
            Default is None, i.e no padding.
        device (torch.device,optional): The device of tensors to which the transform will be applied.
    """

    def __init__(self, size, padding=None, device='cpu'):
        self.size = size
        self.padding = padding
        self.device = device

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor of size (N, C, H, W) to be cropped.
        Returns:
            Tensor: Randomly cropped Tensor.
        """
        if self.padding is not None:
            padded = torch.zeros((tensor.size(0), tensor.size(1), tensor.size(2) + self.padding * 2,
                                  tensor.size(3) + self.padding * 2), dtype=tensor.dtype, device=self.device)
            padded[:, :, self.padding:-self.padding,
                   self.padding:-self.padding] = tensor
        else:
            padded = tensor

        w, h = padded.size(2), padded.size(3)
        th, tw = self.size, self.size
        if w == tw and h == th:
            i, j = 0, 0
        else:
            i = torch.randint(
                0, h - th + 1, (tensor.size(0),), device=self.device)
            j = torch.randint(
                0, w - tw + 1, (tensor.size(0),), device=self.device)

        rows = torch.arange(th, dtype=torch.long,
                            device=self.device) + i[:, None]
        columns = torch.arange(tw, dtype=torch.long,
                               device=self.device) + j[:, None]
        padded = padded.permute(1, 0, 2, 3)
        padded = padded[:, torch.arange(tensor.size(
            0))[:, None, None], rows[:, torch.arange(th)[:, None]], columns[:, None]]
        return padded.permute(1, 0, 2, 3)

def fold_axis(x, y):
    "a util function to fold the x-axis (list) around its center along with its corresponding y values (list) and return the new folded x and y axix "
    if len(x) % 2 == 0:
        raise Exception("uable to fold even length sequence")
    if len(x) != len(y):
        raise ValueError("x and y must have the same length")
    mid = int(np.floor(len(x)/2))
    new_x = []
    new_y = []
    new_x.append(x[mid])
    new_y.append(y[mid])

    for ii in range(mid):
        new_x.append(x[mid+ii+1])
        new_y.append(np.mean((y[mid+ii+1], y[mid-ii-1])))
    return new_x, new_y


def count_verts_faces_trimesh(scene_or_mesh):
    """
    a util function for Trimesh to count the vertices and faces of trimeh object or a cenee of multiple objects
    """
    all_verts = 0
    all_faces = 0
    if isinstance(scene_or_mesh, trimesh.Scene):
        for k, v in mymesh.geometry.items():
            all_verts += v.vertices.data.shape[0]
            all_faces += v.faces.data.shape[0]
    else:
        all_verts = scene_or_mesh.vertices.data.shape[0]
        all_faces = scene_or_mesh.faces.data.shape[0]
    return all_verts, all_faces


def as_mesh(scene_or_mesh):
    """
    Convert a possible scene to a mesh.

    If conversion occurs, the returned mesh has only vertex and face data.
    """
    if isinstance(scene_or_mesh, trimesh.Scene):
        if len(scene_or_mesh.geometry) == 0:
            mesh = None  # empty scene
        else:
            # we lose texture information here
            mesh = trimesh.util.concatenate(
                tuple(trimesh.Trimesh(vertices=g.vertices, faces=g.faces)
                      for g in scene_or_mesh.geometry.values()))
    else:
        assert(isinstance(mesh, trimesh.Trimesh))
        mesh = scene_or_mesh
    return mesh

def chop_ptc(points, factor=0.1, axis=0):
    if factor == 0:
        return points
    new_batch = []
    percentage = 2 * abs(factor) - 1
    for j in range(points.shape[0]):
            idx = np.sign(factor) * points[j][::, axis] > percentage
            new = points[j][idx]
            if new.shape[0] == 0:
                new_batch.append(np.zeros_like(points[j]))
            else:
                new = np.repeat(
                    new, 2 + (points.shape[1]-new.shape[0])/new.shape[0], axis=0)
                new = new[:points.shape[1], ...]
                new_batch.append(new)
    return np.array(new_batch)

def torch_color(color_type,custom_color=(1.0,0,0),max_lightness=False,epsilon=0.00001):
    """
    a function to return a torch tesnor of size 3 that represent a color according to the 'color_type' string that can be [white,red,green,black,random,custom] .. if max_lightness is true , color is normlaized to be brightest
    """
    if color_type == "white":
        color =  torch.tensor((1.0, 1.0, 1.0))
    if color_type == "red":
        color =  torch.tensor((1.0, 0.0, 0.0))
    if color_type == "green":
        color = torch.tensor((0.0, 1.0, 0.0))
    if color_type == "blue":
        color = torch.tensor((0.0, 0.0, 1.0))
    if color_type == "black":
        color = torch.tensor((0.0, 0.0, 0.0))
    elif color_type == "random":
        color = torch.rand(3)
    elif color =="custom":
        color = torch.tensor(custom_color)
    if max_lightness and color_type != "black":
        color = color / (torch.max(color) + epsilon)  # + torch.min(color))
    return color

def save_text(lines, file_name):
    """
    a helper funcion to saves text from the list `lines` and as the file `file_name`
    """
    f = open(file_name, 'w')
    f.writelines(["{}\n".format(x) for x in lines])
    f.close()


def load_text(file_name):
    """
    a helper funcion to load text as lines and return a list of lines without `\n`
    """
    if not os.path.isfile(file_name):
        raise NameError("The file {} does not exisit".format(file_name))
    f = open(file_name, "r")
    lines = f.readlines()
    lines = [line.replace("\n", "") for line in lines]
    f.close()
    return lines



def unit_spherical_grid(nb_points, return_radian=False, return_vertices=False):
    """
    a function that samples a grid of sinze `nb_points` around a sphere of radius `r` . it returns azimth and elevation angels arouns the sphere. if `return_vertices` is true .. it returns the 3d points as well 
    """
    r = 1.0
    vertices = []
    azim = []
    elev = []
    alpha = 4.0*np.pi*r*r/nb_points
    d = np.sqrt(alpha)
    m_nu = int(np.round(np.pi/d))
    d_nu = np.pi/m_nu
    d_phi = alpha/d_nu
    count = 0
    for m in range(0, m_nu):
        nu = np.pi*(m+0.5)/m_nu
        m_phi = int(np.round(2*np.pi*np.sin(nu)/d_phi))
        for n in range(0, m_phi):
            phi = 2*np.pi*n/m_phi
            xp = r*np.sin(nu)*np.cos(phi)
            yp = r*np.sin(nu)*np.sin(phi)
            zp = r*np.cos(nu)
            vertices.append([xp, yp, zp])
            azim.append(phi)
            elev.append(nu-np.pi*0.5)
            count = count + 1
    if not return_radian:
        azim = np.rad2deg(azim)
        elev = np.rad2deg(elev)
    if return_vertices:
        return azim[:nb_points], elev[:nb_points], np.array(vertices[:nb_points])
    else:
        return azim[:nb_points], elev[:nb_points]

def seed_torch(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.enabled = False
def check_valid_rotation_matrix(R, tol: float = 1e-6):
    """
    Determine if R is a valid rotation matrix by checking it satisfies the
    following conditions:
    ``RR^T = I and det(R) = 1``
    Args:
        R: an (N, 3, 3) matrix
    Returns:
        None
    Emits a warning if R is an invalid rotation matrix.
    """
    N = R.shape[0]
    eye = torch.eye(3, dtype=R.dtype, device=R.device)
    eye = eye.view(1, 3, 3).expand(N, -1, -1)
    orthogonal = torch.allclose(R.bmm(R.transpose(1, 2)), eye, atol=tol)
    det_R = torch.det(R)
    no_distortion = torch.allclose(det_R, torch.ones_like(det_R))
    return orthogonal and no_distortion
def zero_nans(tensor):
    """
    zeros all the `nan` values in the pytorch tensor `tensor`
    """
    return torch.where(torch.isnan(tensor), torch.zeros_like(tensor), tensor)

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def get_current_step(optimizer):
    for param_group in optimizer.param_groups:
        for p in param_group['params']:
            return optimizer.state[p]['step']

def torch_center_and_normalize(points,p="inf"):
    """
    a helper pytorch function that normalize and center 3D points clouds 
    """
    N = points.shape[0]
    center = points.mean(0)
    if p != "fro" and p!= "no":
        scale = torch.max(torch.norm(points - center, p=float(p),dim=1))
    elif p=="fro" :
        scale = torch.norm(points - center, p=p )
    elif p=="no":
        scale = 1.0
    points = points - center.expand(N, 3)
    points = points * (1.0 / float(scale))
    return points

def torch_augment_pointcloud(pointcloud):
    """
    for scaling and shifting the point cloud by pytorch
    :param pointcloud:
    :return:
    """
    ## This function takes as input a point cloud of layout `N x 3`,
    ## and output the scaled and shifted point cloud of layout `N x 3`.
    ## hint: useful function `np.random.uniform`, `np.multiply` and `np.add`
    ## TASK 1.1.1 generate a scale variable of size [3] from a uniform distruction between [2/3, 3/2] of size [3]. scale will be used to multiply with the point cloud
    # scale = torch.random.uniform(2.0/3, 3.0/2, 3)
    scale =torch.FloatTensor(3).uniform_(2.0/3, 3.0/2)



    shift = torch.FloatTensor(3).uniform_(-0.2, 0.2)


    ## TASK 1.1.3 scale and then shift the point cloud.
    augmented_pointcloud = shift + pointcloud * scale

    return augmented_pointcloud.to(torch.float)

def logEpoch(logger, model, epoch, loss, accuracy):
    # 1. Log scalar values (scalar summary)
    info = {'loss': loss.item(), 'accuracy': accuracy.item()}

    for tag, value in info.items():
        logger.scalar_summary(tag, value, epoch)

    # 2. Log values and gradients of the parameters (histogram summary)
    for tag, value in model.named_parameters():
        tag = tag.replace('.', '/')
        logger.histo_summary(tag, value.data.cpu().numpy(), epoch)
        logger.histo_summary(tag + '/grad', value.grad.data.cpu().numpy(), epoch)



def rotation_matrix(axis, theta, in_degrees=True):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    if in_degrees:
        theta = math.radians(theta)
    axis = np.asarray(axis)
    axis = axis / math.sqrt(np.dot(axis, axis))
    a = math.cos(theta / 2.0)
    b, c, d = -axis * math.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])


def batch_tensor(tensor, dim=1, squeeze=False):
    """
    a function to reshape pytorch tensor `tensor` along some dimension `dim` to the batch dimension 0 such that the tensor can be processed in parallel. 
    if `sqeeze`=True , the diension `dim` will be removed completelelky, otherwize it will be of size=1.  cehck `unbatch_tensor()` for the reverese function 
    """
    batch_size, dim_size = tensor.shape[0], tensor.shape[dim]
    returned_size = list(tensor.shape)
    returned_size[0] = batch_size*dim_size
    returned_size[dim] = 1
    if squeeze:
        return tensor.transpose(0, dim).reshape(returned_size).squeeze_(dim)
    else:
        return tensor.transpose(0, dim).reshape(returned_size)


def unbatch_tensor(tensor, batch_size, dim=1, unsqueeze=False):
    """
    a function to chunk pytorch tensor `tensor` along the batch dimension 0 and cincatenate the chuncks on dimension `dim` to recover from `batch_tensor()` function.
    if `unsqueee`=True , it will add a dimension `dim` before the unbatching 
    """
    fake_batch_size = tensor.shape[0]
    nb_chunks = int(fake_batch_size / batch_size)
    if unsqueeze:
        return torch.cat(torch.chunk(tensor.unsqueeze_(dim), nb_chunks, dim=0), dim=dim).contiguous()
    else:
        return torch.cat(torch.chunk(tensor, nb_chunks, dim=0), dim=dim).contiguous()



def image_grid(
    images,
    rows=None,
    cols=None,
    fill: bool = True,
    show_axes: bool = False,
    rgb: bool = True,
):
    """
    A util function for plotting a grid of images.

    Args:
        images: (N, H, W, 4) array of RGBA images
        rows: number of rows in the grid
        cols: number of columns in the grid
        fill: boolean indicating if the space between images should be filled
        show_axes: boolean indicating if the axes of the plots should be visible
        rgb: boolean, If True, only RGB channels are plotted.
            If False, only the alpha channel is plotted.

    Returns:
        None
    """
    if (rows is None) != (cols is None):
        raise ValueError("Specify either both rows and cols or neither.")

    if rows is None:
        rows = len(images)
        cols = 1

    gridspec_kw = {"wspace": 0.0, "hspace": 0.0} if fill else {}
    fig, axarr = plt.subplots(
        rows, cols, gridspec_kw=gridspec_kw, figsize=(15, 9))
    bleed = 0
    fig.subplots_adjust(left=bleed, bottom=bleed,
                        right=(1 - bleed), top=(1 - bleed))

    for ax, im in zip(axarr.ravel(), images):
        if rgb:
            # only render RGB channels
            ax.imshow(im[..., :3])
        else:
            # only render Alpha channel
            ax.imshow(im[..., 3])
        if not show_axes:
            ax.set_axis_off()


from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import


def get_camera_wireframe(scale: float = 0.3):
    """
    Returns a wireframe of a 3D line-plot of a camera symbol.
    """
    a = 0.5 * torch.tensor([-2, 1.5, 4])
    b = 0.5 * torch.tensor([2, 1.5, 4])
    c = 0.5 * torch.tensor([-2, -1.5, 4])
    d = 0.5 * torch.tensor([2, -1.5, 4])
    C = torch.zeros(3)
    F = torch.tensor([0, 0, 3])
    camera_points = [a, b, d, c, a, C, b, d, C, c, C, F]
    lines = torch.stack([x.float() for x in camera_points]) * scale
    return lines


def plot_cameras(ax, cameras, color: str = "blue", scale: float = 0.3):
    """
    Plots a set of `cameras` objects into the maplotlib axis `ax` with
    color `color`.
    """
    cam_wires_canonical = get_camera_wireframe(scale).cuda()[None]
    cam_trans = cameras.get_world_to_view_transform().inverse()
    cam_wires_trans = cam_trans.transform_points(cam_wires_canonical)
    plot_handles = []
    for wire in cam_wires_trans:
        # the Z and Y axes are flipped intentionally here!
        x_, z_, y_ = wire.detach().cpu().numpy().T.astype(float)
        (h,) = ax.plot(x_, y_, z_, color=color, linewidth=0.3)
        plot_handles.append(h)
    return plot_handles


def plot_camera_scene(cameras, cameras_gt, status: str):
    """
    Plots a set of predicted cameras `cameras` and their corresponding
    ground truth locations `cameras_gt`. The plot is named with
    a string passed inside the `status` argument.
    """
    fig = plt.figure()
    ax = fig.gca(projection="3d")
    ax.clear()
    ax.set_title(status)
    handle_cam = plot_cameras(ax, cameras, color="#FF7D1E")
    handle_cam_gt = plot_cameras(ax, cameras_gt, color="#812CE5")
    plot_radius = 3
    ax.set_xlim3d([-plot_radius, plot_radius])
    ax.set_ylim3d([3 - plot_radius, 3 + plot_radius])
    ax.set_zlim3d([-plot_radius, plot_radius])
    ax.set_xlabel("x")
    ax.set_ylabel("z")
    ax.set_zlabel("y")
    labels_handles = {
        "Estimated cameras": handle_cam[0],
        "GT cameras": handle_cam_gt[0],
    }
    ax.legend(
        labels_handles.values(),
        labels_handles.keys(),
        loc="upper center",
        bbox_to_anchor=(0.5, 0),
    )
    plt.show()
    return fig


def save_cameras(cameras, save_path, scale=0.22, dpi=200):
    import mpl_toolkits
    fig = plt.figure()
    ax = mpl_toolkits.mplot3d.axes3d.Axes3D(fig)
    ax.set_xlim(-3.0, 3.0)
    ax.set_ylim(-3.0, 3.0)
    ax.set_zlim(-1.8, 1.8)
    ax.scatter(xs=[0], ys=[0], zs=[0], linewidth=3, c="r")
    plot_cameras(ax, cameras, color="blue", scale=scale)
    plt.savefig(save_path, dpi=dpi)
    plt.close(fig)


def save_grid(image_batch, save_path, **kwargs):
    """
    a hleper function for torchvision.util function `make_grid` to save a batch of images (B,H,W,C) as a grid on the `save_path` 
    """
    from torchvision.utils import make_grid
    im = make_grid(image_batch, **kwargs).detach().cpu().transpose(0, 2).transpose(0, 1).numpy()
    imageio.imsave(save_path, (255.0*im).astype(np.uint8))


def save_obj(obj, name):
    with open(name, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    with open(name, 'rb') as f:
        return pickle.load(f, encoding='latin1')


def devide_points_by_labels(points, labels):
    """
    divides the point clouds `points` as numpy array into a list of point clouds each belonging to one of the labels in `labels` 
    """
    return [points[labels == lbl] for lbl in np.unique(labels)]


def rotation_matrix(axis, theta, in_degrees=True):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    if in_degrees:
        theta = math.radians(theta)
    axis = np.asarray(axis)
    axis = axis / math.sqrt(np.dot(axis, axis))
    a = math.cos(theta / 2.0)
    b, c, d = -axis * math.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])


def sort_jointly(list_of_arrays, dim=0):
    """
    sort all the arrays in `list_of_arrays` according to the sorting of the array `array list_of_arrays`[dim]
    """
    def swapPositions(mylsit, pos1, pos2):
        mylsit[pos1], mylsit[pos2] = mylsit[pos2], mylsit[pos1]
        return mylsit
    sorted_tuples = sorted(zip(*swapPositions(list_of_arrays, 0, dim)))
    combined_sorted = list(zip(*sorted_tuples))
    return [list(ii) for ii in swapPositions(combined_sorted, 0, dim)]


def diff_1d(array):
    """
    computes the 1d gradient of the array 
    """
    new_array = np.zeros(len(array) + 1)
    new_array[:len(array)] = array
    return np.array([(new_array[ii+1] - new_array[ii]) for ii in range(len(array))])


def dict_dict_to_matrix(mydict, order=None):
    """
    converts a dict of dict to matrix of size len(mydict) * len(mydict) with the order of keys dictated by the list order 
    """
    if not order:
        keys = mydict.keys()
    else:
        keys = order
    result_matrix = []
    for k in keys:
        result_matrix.append([mydict[k][kk] for kk in keys])
    return np.array(result_matrix)


def gif_folder(data_dir, extension="jpg", duration=None):
    """
    converts a folder of images in `data_dir` into a gif named `animation.gif` in the same directory  
    """
    image_collection = []
    for img_name in sorted(glob.glob(data_dir + "/*." + extension)):
        image_collection.append(imageio.imread(img_name))
    if not duration:
        imageio.mimsave(os.path.join(
            data_dir, "animation.gif"), image_collection)
    else:
        imageio.mimsave(os.path.join(data_dir, "animation.gif"),
                        image_collection, duration=duration)
# mesh = PyntCloud.from_file("/media/hamdiaj/D/mywork/sublime/vgd/3d/ModelNet40/airplane/test/airplane_0627.off")


def check_folder(data_dir):
    """
    checks if folder exists and create if doesnt exist 
    """
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)




def random_id(digits_nb=4, include_letters=True, only_capital=True, unique_digits=False):
    """
    generates random digits of length digits_nb and return the random string  . options include using letters or only numbers , unique or not uniqe digits, all capital letters or allow lowercase 
    
    Args:
        digits_nb : (int) the number of didits to be returned in the resulting 
        include_letters : (bool) flag wheathre to include letters or only numbers in the string 
        only_capital : (bool) flag wheathre to allow lowercase letters if include_letters==True
        unique_digits : (bool) flag wheathre to make all the digits unique or allow repetetion 
    Returns :
        string of random digits 
    """
    import numpy as np
    if include_letters:
        full_list = list(
            '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ')
        if only_capital:
            full_list = list('0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ')
    else:
        full_list = list('0123456789')
    short_list = np.random.choice(
        full_list, digits_nb, replace=not unique_digits)
    mystr = ""
    for ii in short_list:
        mystr = mystr+ii
    return mystr


def combine_csvs(files_list, outfile):
    df_list = []
#     print(file_list)
    for file_name in files_list:
        df = pd.read_csv(file_name)
        df_list.append(df)
    df_out = pd.concat(df_list, sort=False)
    df_out.to_csv(outfile, index=False, sep=",")
    return df_out


def merge_two_dicts(x, y):
    z = x.copy()   # start with x's keys and values
    z.update(y)    # modifies z with y's keys and values & returns None
    return z

class ListDict(object):
    """
    a class of list dictionary .. each element is a list , has the methods of both lists and dictionaries 
    idel for combining the results of some experimtns and setups 
    """

    def __init__(self, keylist_or_dict=None):
        # def initilize_list_dict(names):
        if isinstance(keylist_or_dict, list):
            self.listdict = {k: [] for k in keylist_or_dict}
        elif isinstance(keylist_or_dict, dict):
            if isinstance(list(keylist_or_dict.values())[0], list):
                self.listdict = copy.deepcopy(keylist_or_dict)
            else:
                self.listdict = {k: [v] for k, v in keylist_or_dict.items()}
        elif isinstance(keylist_or_dict, ListDict):
            self.listdict = copy.deepcopy(keylist_or_dict)
        elif not keylist_or_dict:
            self.listdict = {}
        else:
            print("unkonwn type")

    def raw_dict(self):
        """
        returns the Dict object that is iassoicaited with the ListDict object 
        """
        return self.listdict

    def append(self, one_dict):
        if len(self.listdict) == 0:
            self.listdict = {k: [v] for k, v in one_dict.items()}
        else:
            for k, v in self.items():
                v.append(one_dict[k])
        return self

    def extend(self, newlistdict):
        if len(self.listdict) == 0:
            self.listdict = newlistdict.listdict
        else:
            for k, v in self.items():
                v.extend(newlistdict.raw_dict()[k])
        return self

    def partial_append(self, one_dict):
        for k, v in one_dict.items():
            self.listdict[k].append(v)
        return self

    def partial_extend(self, newlistdict):
        for k, v in newlistdict.items():
            self.listdict[k].extend(v)
        return self

    def __add__(self, newlistdict):
        return ListDict(merge_two_dicts(self.raw_dict(), newlistdict.raw_dict()))

    def combine(self, newlistdict):
        self.listdict = merge_two_dicts(
            self.raw_dict(), newlistdict.raw_dict())
        # self.listdict = {**self.raw_dict(), **newlistdict.raw_dict()}
        return self

    def __sub__(self, newlistdict):
        new_dict = ListDict(self.raw_dict())
        for k in newlistdict.raw_dict().keys():
            new_dict.raw_dict().pop(k, None)
        return new_dict

    def remove(self, newlistdict):
        for k in newlistdict.raw_dict().keys():
            self.listdict.pop(k, None)
        return self

    def chek_error(self):
        for k, v in self.items():
            print(len(v), ":", k)
        return self

    def __getitem__(self, key):
        return self.listdict[key]

    def __str__(self):
        return str(self.listdict)

    def __len__(self):
        return len(self.listdict)

    def keys(self):
        return self.listdict.keys()

    def values(self):
        return self.listdict.values()

    def items(self):
        return self.listdict.items()
    # def save(self,save_file):
    #     raise NotImplementedError

def log_setup(setup, setups_file):
    """
    update an exisiting CSV file or create new one if not exisiting using setup
    """
    setup_ld = ListDict(setup)
    if os.path.isfile(setups_file):
        old_ld = ListDict(pd.read_csv(setups_file, sep=",").to_dict("list"))
        old_ld.append(setup)
        setup_ld = old_ld
    pd.DataFrame(setup_ld.raw_dict()).to_csv(setups_file, sep=",", index=False)


def save_results(save_file, results):
    pd.DataFrame(results.raw_dict()).to_csv(save_file, sep=",", index=False)


def load_results(load_file):
    if os.path.isfile(load_file):
        df = pd.read_csv(load_file, sep=",")
        return ListDict(df.to_dict("list"))
    else:
        print(" ########## WARNING : no file names : {}".format(load_file))
        return None


def down_sample_ptc(points, target):
    """
    downsamples a numpy point cloud `points` to a `target` number of points
    """
    if target > points.shape[0]:
        return points
    import random
    return np.array(random.sample(list(points), target))


def down_sample_ptc_batch(points_batch, target):
    """
    downsamples a batch of numpy point clouds `points_batch` to a `target` number of points
    """
    down_sampled_batch = []
    for ii in range(points_batch.shape[0]):
        down_sampled_batch.append(
            down_sample_ptc(points_batch[ii, ...], target))
    return np.array(down_sampled_batch)


def up_sample_ptc(points, target):
    return np.repeat(points, target, axis=0)


def up_sample_ptc_batch(points_batch, target):
    up_sampled_batch = []
    for ii in range(points_batch.shape[0]):
        up_sampled_batch.append(up_sample_ptc(points_batch[ii, ...], target))
    return np.array(up_sampled_batch)

