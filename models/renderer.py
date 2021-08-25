import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from util import *
from ops import check_and_correct_rotation_matrix
import torch
from torch.autograd import Variable
import numpy as np
# to import files from parent dir

from torch import nn
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
from pytorch3d.renderer.cameras import camera_position_from_spherical_angles

ORTHOGONAL_THRESHOLD = 1e-6
EXAHSTION_LIMIT = 20



class MVRenderer(nn.Module):
    """
    The Multi-view differntiable renderer main class that render multiple views differntiably from some given viewpoints. It can render meshes and point clouds as well
    Args: 
        `nb_views` int , The number of views used in the multi-view setup
        `image_size` int , The image sizes of the rendered views.
        `pc_rendering` : bool , flag to use point cloud rendering instead of mesh rendering
        `object_color` : str , The color setup of the objects/points rendered. Choices: ["white", "random","black","red","green","blue", "custom"]
        `background_color` : str , The color setup of the rendering background. Choices: ["white", "random","black","red","green","blue", "custom"]
        `faces_per_pixel` int , The number of faces rendered per pixel when mesh rendering is used (`pc_rendering` == `False`) .
        `points_radius`: float , the radius of the points rendered. The more points in a specific `image_size`, the less radius required for proper rendering.
        `points_per_pixel` int , The number of points rendered per pixel when point cloud rendering is used (`pc_rendering` == `True`) .
        `light_direction` : str , The setup of the light used in rendering when mesh rendering is available. Choices: ["fixed", "random", "relative"]
        `cull_backfaces` : bool , Allow backface-culling when rendering meshes (`pc_rendering` == `False`).

    Returns:
        an MVTN object that can render multiple views according to predefined setup
    """

    def __init__(self, nb_views, image_size=224, pc_rendering=True, object_color="white", background_color="white", faces_per_pixel=1, points_radius=0.006,  points_per_pixel=1, light_direction="random", cull_backfaces=False):
        super().__init__()
        self.nb_views = nb_views
        self.image_size = image_size
        self.pc_rendering = pc_rendering
        self.object_color = object_color
        self.background_color = background_color
        self.faces_per_pixel = faces_per_pixel
        self.points_radius = points_radius
        self.points_per_pixel = points_per_pixel
        self.light_direction_type = light_direction
        self.cull_backfaces = cull_backfaces

        
        # self.EPSILON = 0.00001 # color normalization epsilon 

    def render_meshes(self,meshes, color, azim, elev, dist, lights, background_color=(1.0, 1.0, 1.0), ):
        c_batch_size = len(meshes)
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
            verts_rgb=color.cuda()*torch.ones((c_batch_size, max_vert, 3)).cuda())
        # Create a Meshes object for the teapot. Here we have only one mesh in the batch.
        R, T = look_at_view_transform(dist=batch_tensor(dist.T, dim=1, squeeze=True), elev=batch_tensor(
            elev.T, dim=1, squeeze=True), azim=batch_tensor(azim.T, dim=1, squeeze=True))
        R, T = check_and_correct_rotation_matrix(
            R, T, EXAHSTION_LIMIT, azim, elev, dist)

        cameras = OpenGLPerspectiveCameras(
            device="cuda:{}".format(torch.cuda.current_device()), R=R, T=T)
        camera = OpenGLPerspectiveCameras(device="cuda:{}".format(torch.cuda.current_device()), R=R[None, 0, ...],
                                        T=T[None, 0, ...])

        # camera2 = OpenGLPerspectiveCameras(device=device, R=R[None, 2, ...],T=T[None, 2, ...])
        # print(camera2.get_camera_center())
        raster_settings = RasterizationSettings(
            image_size=self.image_size,
            blur_radius=0.0,
            faces_per_pixel=self.faces_per_pixel,
            # bin_size=None, #int
            # max_faces_per_bin=None,  # int
            # perspective_correct=False,
            # clip_barycentric_coords=None, #bool
            cull_backfaces=self.cull_backfaces,
        )
        renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=camera, raster_settings=raster_settings),
            shader=HardPhongShader(blend_params=BlendParams(background_color=background_color
                                                            ), device=lights.device, cameras=camera, lights=lights)
        )
        new_meshes = new_meshes.extend(self.nb_views)

        # compute output
        # print("after rendering .. ", rendered_images.shape)

        rendered_images = renderer(new_meshes, cameras=cameras, lights=lights)

        rendered_images = unbatch_tensor(
            rendered_images, batch_size=self.nb_views, dim=1, unsqueeze=True).transpose(0, 1)
        # print(rendered_images[:, 100, 100, 0])

        rendered_images = rendered_images[..., 0:3].transpose(2, 4).transpose(3, 4)
        return rendered_images, cameras

    def render_points(self, points, color, azim, elev, dist, background_color=(0.0, 0.0, 0.0), ):
        c_batch_size = azim.shape[0]

        point_cloud = Pointclouds(points=points.to(torch.float), features=color *
                                torch.ones_like(points, dtype=torch.float)).cuda()

        # print(len(new_meshes.faces_list()[0]))
        # Create a Meshes object for the teapot. Here we have only one mesh in the batch.
        R, T = look_at_view_transform(dist=batch_tensor(dist.T, dim=1, squeeze=True), elev=batch_tensor(
            elev.T, dim=1, squeeze=True), azim=batch_tensor(azim.T, dim=1, squeeze=True))
        R, T = check_and_correct_rotation_matrix(
            R, T, EXAHSTION_LIMIT, azim, elev, dist)

        cameras = OpenGLOrthographicCameras(device="cuda:{}".format(
            torch.cuda.current_device()), R=R, T=T, znear=0.01)
        raster_settings = PointsRasterizationSettings(
            image_size=self.image_size,
            radius=self.points_radius,
            points_per_pixel=self.points_per_pixel
        )

        renderer = PointsRenderer(
            rasterizer=PointsRasterizer(
                cameras=cameras, raster_settings=raster_settings),
            compositor=NormWeightedCompositor(background_color=background_color)
        )
        point_cloud = point_cloud.extend(self.nb_views)
        point_cloud.scale_(batch_tensor(1.0/dist.T, dim=1,
                                        squeeze=True)[..., None][..., None])

        rendered_images = renderer(point_cloud)
        rendered_images = unbatch_tensor(
            rendered_images, batch_size=self.nb_views, dim=1, unsqueeze=True).transpose(0, 1)

        rendered_images = rendered_images[..., 0:3].transpose(2, 4).transpose(3, 4)
        return rendered_images, cameras

    def rendering_color(self, custom_color=(1.0, 0, 0)):
        if self.object_color == "custom":
            color =  custom_color
        elif self.object_color == "random" and not self.training:
            color = torch_color("white")
        else:
            color = torch_color(self.object_color,max_lightness=True,)
        return color

    def light_direction(self, azim, elev, dist):
        if self.light_direction_type == "fixed":
            return ((0, 1.0, 0),)
        elif self.light_direction_type == "random" and self.training:
            return (tuple(1.0 - 2 * np.random.rand(3)),)
        else:  
         relative_view = Variable(camera_position_from_spherical_angles(distance=batch_tensor(dist.T, dim=1, squeeze=True), elevation=batch_tensor(
             elev.T, dim=1, squeeze=True), azimuth=batch_tensor(azim.T, dim=1, squeeze=True))).to(torch.float)
        #  return correction_factor.repeat_interleave((self.nb_views))[..., None].repeat(1, 3).to(torch.float) * relative_view
        return relative_view

    def forward(self, meshes, points, azim, elev, dist, color=None):
        """
        The main rendering function of the MVRenderer class. It can render meshes (if `self.pc_rendering` == `False`) or 3D point clouds(if `self.pc_rendering` == `True`).
        Arge:
            `meshes`: a list of B `Pytorch3D.Mesh` to be rendered , B batch size. In case not available, just pass `None`. 
            `points`: B * N * 3 tensor, a batch of B point clouds to be rendered where each point cloud has N points and each point has X,Y,Z property. In case not available, just pass `None` .
            `azim`: B * M tensor, a B batch of M azimth angles that represent the azimth angles of the M view-points to render the points or meshes from.
            `elev`: B * M tensor, a B batch of M elevation angles that represent the elevation angles of the M view-points to render the points or meshes from.
            `dist`:  B * M tensor, a B batch of M unit distances that represent the distances of the M view-points to render the points or meshes from.
            `color`: B * N * 3 tensor, The RGB colors of batch of point clouds/meshes with N is the number of points/vertices  and B batch size. Only if `self.object_color` == `custom`, otherwise this option not used

        """
        background_color = torch_color(self.background_color, max_lightness=True,).cuda()
        color = self.rendering_color(color)

        if not self.pc_rendering:
            lights = DirectionalLights(
                device=background_color.device, direction=self.light_direction(azim, elev, dist))

            rendered_images, cameras = self.render_meshes(
                meshes=meshes, color=color, azim=azim, elev=elev, dist=dist, lights=lights, background_color=background_color)
        else:
            rendered_images, cameras = self.render_points(
                points=points, color=color, azim=azim, elev=elev, dist=dist, background_color=background_color)
        return rendered_images, cameras

    def render_and_save(self, meshes, points, azim, elev, dist, images_path, cameras_path, color=None):
        with torch.no_grad():
            rendered_images, cameras = self.forward(meshes, points, azim, elev, dist, color)
        # print("before saving .. ",rendered_images.shape)
        save_grid(image_batch=rendered_images[0, ...],
                save_path=images_path, nrow=self.nb_views)
        save_cameras(cameras, save_path=cameras_path, scale=0.22, dpi=200)
