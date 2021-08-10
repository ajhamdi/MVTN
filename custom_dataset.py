from typing import Dict, List, Optional, Tuple
from pathlib import Path
from os import path
import warnings
import json
from torch.utils.data import Dataset
import numpy as np
import h5py
from torch.utils.data.dataset import Dataset
import os
from PIL import Image
from util import * 
import re
from torch._six import container_abcs, string_classes, int_classes
# from torch_geometric.io import read_off, read_obj
import trimesh
import math
from pytorch3d.structures import Meshes
from pytorch3d.renderer.mesh import Textures

# 3D transformations functions
from pytorch3d.transforms import Rotate, Translate

# rendering components
from pytorch3d.renderer import (
    OpenGLPerspectiveCameras, look_at_view_transform, look_at_rotation,
    RasterizationSettings, MeshRenderer, MeshRasterizer, BlendParams,
    SoftSilhouetteShader, HardPhongShader, PointLights)

def  rotation_matrix(axis, theta, in_degrees=True):
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
class MultiViewDataSet(Dataset):

    def find_classes(self, dir):
        classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}

        return classes, class_to_idx

    def __init__(self, root, data_type, transform=None, target_transform=None):
        self.x = []
        self.y = []
        self.root = root

        self.classes, self.class_to_idx = self.find_classes(root)

        self.transform = transform
        self.target_transform = target_transform

        # root / <label>  / <train/test> / <item> / <view>.png
        for label in os.listdir(root): # Label
            for item in os.listdir(root + '/' + label + '/' + data_type):
                views = []
                for view in os.listdir(root + '/' + label + '/' + data_type + '/' + item):
                    views.append(root + '/' + label + '/' + data_type + '/' + item + '/' + view)

                self.x.append(views)
                self.y.append(self.class_to_idx[label])

    # Override to give PyTorch access to any image on the dataset
    def __getitem__(self, index):
        orginal_views = self.x[index]
        views = []

        for view in orginal_views:
            im = Image.open(view)
            im = im.convert('RGB')
            if self.transform is not None:
                im = self.transform(im)
            views.append(im)

        return views, self.y[index]

    # Override to give PyTorch size of dataset
    def __len__(self):
        return len(self.x)


class ThreeMultiViewDataSet(Dataset):

    def find_classes(self, dir):
        classes = [d for d in os.listdir(
            dir) if os.path.isdir(os.path.join(dir, d))]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}

        return classes, class_to_idx

    def __init__(self, data_type, setup, transform=None, is_rotated=False):
        # self.x = []
        self.y = []
        self.meshes_list =[]
        self.data_type = data_type
        self.nb_points = setup["nb_points"]
        self.root_2D = setup["image_data"]
        self.root_3D = setup["mesh_data"]
        self.initial_angle = setup["initial_angle"]
        self.simplified_mesh = setup["simplified_mesh"]
        self.cleaned_mesh = setup["cleaned_mesh"]
        self.dset_norm = setup["dset_norm"]
        self.return_points_sampled = setup["return_points_sampled"]
        self.return_points_saved = setup["return_points_saved"]
        self.return_extracted_features = setup["return_extracted_features"]
        self.features_type = setup["features_type"]

        self.classes, self.class_to_idx = self.find_classes(self.root_2D)

        self.transform = transform
        self.is_rotated = is_rotated

        # root / <label>  / <train/test> / <item> / <view>.png
        for label in os.listdir(self.root_2D):  # Label
            for item in os.listdir(self.root_2D + '/' + label + '/' + self.data_type):
                views = []
                for view in os.listdir(self.root_2D + '/' + label + '/' + self.data_type + '/' + item):
                    views.append(self.root_2D + '/' + label + '/' +
                                 self.data_type + '/' + item + '/' + view)

                # self.x.append(views)
                self.y.append(self.class_to_idx[label])
                self.meshes_list.append(
                    self.root_3D + '/' + label + '/' + self.data_type + '/' + item)

        self.simplified_meshes_list = [file_name.replace(
            ".off", "_SMPLER.obj") for file_name in self.meshes_list if file_name[-4::]==".off"]
        # self.cleaned_meshes_list = [file_name.replace(
        #     ".off", "_RMV.obj") for file_name in self.meshes_list if file_name[-4::]==".off"]
        self.extracted_features_list = [file_name.replace(".off", "_PFeautres.pkl") for file_name in self.meshes_list if file_name[-4::] == ".off"]
        self.points_list = [file_name.replace(".off", "POINTS.pkl") for file_name in self.meshes_list if file_name[-4::] == ".off"]
        self.meshes_list, self.simplified_meshes_list, self.y, self.points_list, self.extracted_features_list = sort_jointly(
            [self.meshes_list, self.simplified_meshes_list, self.y, self.points_list, self.extracted_features_list], dim=0)
        if self.is_rotated:
            df = pd.read_csv(os.path.join(self.root_3D,"..", "rotated_modelnet_{}.csv".format(self.data_type)), sep=",")
            self.rotations_list = [df[df.mesh_path.isin([x])].to_dict("list") for x in self.meshes_list]
        
        self.correction_factors = [1.0]*len(self.meshes_list)
        if self.cleaned_mesh:
            fault_mesh_list = load_text(os.path.join(self.root_3D, "..", "{}_faults.txt".format(self.data_type)))
            fault_mesh_list = [int(x) for x in fault_mesh_list]
            for x in fault_mesh_list:
                self.correction_factors[x] = -1.0
        
            # print("@@@@@@@@@@@", self.rotations_list[0], self.meshes_list[0])


    # Override to give PyTorch access to any image on the dataset
    def __getitem__(self, index):
        # orginal_views = self.x[index]
        # views = []

        # for view in orginal_views:
        #     im = Image.open(view)
        #     im = im.convert('RGB')
        #     if self.transform is not None:
        #         im = self.transform(im)
        #     views.append(im)
        if not self.simplified_mesh :
            threeobject = trimesh.load(self.meshes_list[index])
            # threeobject = read_off(self.meshes_list[index])
            # verts = threeobject.pos.numpy()
            # faces = threeobject.face.transpose(0, 1).numpy()
        else:
            threeobject = trimesh.load(self.simplified_meshes_list[index])
        

        if not self.is_rotated:
            angle = self.initial_angle
            rot_axis = [1, 0, 0]
        else :
            angle = self.rotations_list[index]["rot_theta"][0]
            rot_axis = [self.rotations_list[index]["rot_x"]
                        [0], self.rotations_list[index]["rot_y"][0], self.rotations_list[index]["rot_z"][0]]

        verts = np.array(threeobject.vertices.data.tolist())
        faces = np.array(threeobject.faces.data.tolist())
        verts = rotation_matrix(rot_axis, angle).dot(verts.T).T
        verts = torch_center_and_normalize(torch.from_numpy(
            verts).to(torch.float), p=self.dset_norm)
        faces = torch.from_numpy(faces)

        # if True:
        #     faces[ :, 1], faces[ :, 2] = faces[ :, 2], faces[ :, 1]



        # Initialize each vertex to be white in color.
        verts_rgb = torch.ones_like(verts)[None]  # (1, V, 3)
        textures = Textures(verts_rgb=verts_rgb)
        mesh = Meshes(
            verts=[verts],
            faces=[faces],
            textures=textures
        )
        if self.return_points_sampled or self.return_points_saved:
            if self.return_points_sampled:
                points = threeobject.sample(self.nb_points, False)
            else :
                points = load_obj(self.points_list[index])
            points = torch.from_numpy(rotation_matrix(rot_axis, angle).dot(points.T).T).to(torch.float)
            points = torch_center_and_normalize(points, p=self.dset_norm)
            # if self.data_type =="train":
            #     points = torch_augment_pointcloud(points)
            return self.y[index], mesh, points, self.correction_factors[index]
        elif self.return_extracted_features:
            features = load_obj(self.extracted_features_list[index])
            if self.features_type == "logits_trans":
                features = np.concatenate((features["logits"].reshape(-1), features["transform_matrix"].reshape(-1)),0)
            elif self.features_type == "post_max_trans":
                features = np.concatenate(
                    (features["post_max"].reshape(-1), features["transform_matrix"].reshape(-1)), 0)
            else :
                features = features[self.features_type].reshape(-1)

            # if self.data_type =="train":
            #     points = torch_augment_pointcloud(points)
            # print(features[self.features_type].shape)
            return self.y[index], mesh, features, self.correction_factors[index]
        else :
            return self.y[index], mesh, None, self.correction_factors[index]

    # Override to give PyTorch size of dataset
    def __len__(self):
        return len(self.y)



def collate_fn(batch):
    r"""Puts each data field into a tensor with outer dimension batch size"""

    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, torch.Tensor):
        out = None
        if torch.utils.data.get_worker_info() is not None:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum([x.numel() for x in batch])
            storage = elem.storage()._new_shared(numel)
            out = elem.new(storage)
        return torch.stack(batch, 0, out=out)
    elif elem_type.__module__ == 'pytorch3d.structures.meshes':
        return batch
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        if elem_type.__name__ == 'ndarray' or elem_type.__name__ == 'memmap':
            return collate_fn([torch.as_tensor(b) for b in batch])
        elif elem.shape == ():  # scalars
            return torch.as_tensor(batch)
    elif isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float64)
    elif isinstance(elem, int_classes):
        return torch.tensor(batch)
    elif isinstance(elem, string_classes):
        return batch
    elif isinstance(elem, container_abcs.Mapping):
        return {key: collate_fn([d[key] for d in batch]) for key in elem}
    elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
        return elem_type(*(collate_fn(samples) for samples in zip(*batch)))
    elif isinstance(elem, container_abcs.Sequence):
        # check to make sure that the elements in batch have consistent size
        it = iter(batch)
        elem_size = len(next(it))
        if not all(len(elem) == elem_size for elem in it):
            raise RuntimeError(
                'each element in list of batch should be of equal size')
        transposed = zip(*batch)
        return [collate_fn(samples) for samples in transposed]


class ShapeNetBase(torch.utils.data.Dataset):
    """
    'ShapeNetBase' implements a base Dataset for ShapeNet and R2N2 with helper methods.
    It is not intended to be used on its own as a Dataset for a Dataloader. Both __init__
    and __getitem__ need to be implemented.
    """

    def __init__(self):
        """
        Set up lists of synset_ids and model_ids.
        """
        self.synset_ids = []
        self.model_ids = []
        self.synset_inv = {}
        self.synset_start_idxs = {}
        self.synset_num_models = {}
        self.shapenet_dir = ""
        self.model_dir = "model.obj"
        self.load_textures = True
        self.texture_resolution = 4

    def __len__(self):
        """
        Return number of total models in the loaded dataset.
        """
        return len(self.model_ids)

    def __getitem__(self, idx) -> Dict:
        """
        Read a model by the given index. Need to be implemented for every child class
        of ShapeNetBase.

        Args:
            idx: The idx of the model to be retrieved in the dataset.

        Returns:
            dictionary containing information about the model.
        """
        raise NotImplementedError(
            "__getitem__ should be implemented in the child class of ShapeNetBase"
        )

    def _get_item_ids(self, idx) -> Dict:
        """
        Read a model by the given index.

        Args:
            idx: The idx of the model to be retrieved in the dataset.

        Returns:
            dictionary with following keys:
            - synset_id (str): synset id
            - model_id (str): model id
        """
        model = {}
        model["synset_id"] = self.synset_ids[idx]
        model["model_id"] = self.model_ids[idx]
        return model

    def _load_mesh(self, model_path) -> Tuple:
        from pytorch3d.io import load_obj

        verts, faces, aux = load_obj(
            model_path,
            create_texture_atlas=self.load_textures,
            load_textures=self.load_textures,
            texture_atlas_size=self.texture_resolution,
        )
        if self.load_textures:
            textures = aux.texture_atlas
            # Some meshes don't have textures. In this case
            # create a white texture map
        else:
            textures = verts.new_ones(
                faces.verts_idx.shape[0],
                self.texture_resolution,
                self.texture_resolution,
                3,
            )

        return verts, faces.verts_idx, textures

  
class ShapeNetCore(ShapeNetBase):
    """
    This class loads ShapeNetCore from a given directory into a Dataset object.
    ShapeNetCore is a subset of the ShapeNet dataset and can be downloaded from
    https://www.shapenet.org/.
    """

    def __init__(
        self,
        data_dir,
        split,
        nb_points,
        synsets=None,
        version: int = 2,
        load_textures: bool = False,
        texture_resolution: int = 4,
        dset_norm: str = "inf",
        simplified_mesh=False
    ):
        """
        Store each object's synset id and models id from data_dir.
        Args:
            data_dir: Path to ShapeNetCore data.
            synsets: List of synset categories to load from ShapeNetCore in the form of
                synset offsets or labels. A combination of both is also accepted.
                When no category is specified, all categories in data_dir are loaded.
            version: (int) version of ShapeNetCore data in data_dir, 1 or 2.
                Default is set to be 1. Version 1 has 57 categories and verions 2 has 55
                categories.
                Note: version 1 has two categories 02858304(boat) and 02992529(cellphone)
                that are hyponyms of categories 04530566(watercraft) and 04401088(telephone)
                respectively. You can combine the categories manually if needed.
                Version 2 doesn't have 02858304(boat) or 02834778(bicycle) compared to
                version 1.
            load_textures: Boolean indicating whether textures should loaded for the model.
                Textures will be of type TexturesAtlas i.e. a texture map per face.
            texture_resolution: Int specifying the resolution of the texture map per face
                created using the textures in the obj file. A
                (texture_resolution, texture_resolution, 3) map is created per face.
        """
        super().__init__()
        self.shapenet_dir = data_dir
        self.nb_points = nb_points
        self.load_textures = load_textures
        self.texture_resolution = texture_resolution
        self.dset_norm = dset_norm
        self.split = split
        self.simplified_mesh = simplified_mesh

        if version not in [1, 2]:
            raise ValueError("Version number must be either 1 or 2.")
        self.model_dir = "model.obj" if version == 1 else "models/model_normalized.obj"
        if self.simplified_mesh:
            self.model_dir = "models/model_normalized_SMPLER.obj"
        splits = pd.read_csv(os.path.join(
            self.shapenet_dir, "shapenet_split.csv"), sep=",", dtype=str)

        # Synset dictionary mapping synset offsets to corresponding labels.
        dict_file = "shapenet_synset_dict_v%d.json" % version
        with open(path.join(self.shapenet_dir, dict_file), "r") as read_dict:
            self.synset_dict = json.load(read_dict)
        # Inverse dicitonary mapping synset labels to corresponding offsets.
        self.synset_inv = {label: offset for offset,
                           label in self.synset_dict.items()}

        # If categories are specified, check if each category is in the form of either
        # synset offset or synset label, and if the category exists in the given directory.
        if synsets is not None:
            # Set of categories to load in the form of synset offsets.
            synset_set = set()
            for synset in synsets:
                if (synset in self.synset_dict.keys()) and (
                    path.isdir(path.join(data_dir, synset))
                ):
                    synset_set.add(synset)
                elif (synset in self.synset_inv.keys()) and (
                    (path.isdir(path.join(data_dir, self.synset_inv[synset])))
                ):
                    synset_set.add(self.synset_inv[synset])
                else:
                    msg = (
                        "Synset category %s either not part of ShapeNetCore dataset "
                        "or cannot be found in %s."
                    ) % (synset, data_dir)
                    warnings.warn(msg)
        # If no category is given, load every category in the given directory.
        # Ignore synset folders not included in the official mapping.
        else:
            synset_set = {
                synset
                for synset in os.listdir(data_dir)
                if path.isdir(path.join(data_dir, synset))
                and synset in self.synset_dict
            }

        # Check if there are any categories in the official mapping that are not loaded.
        # Update self.synset_inv so that it only includes the loaded categories.
        synset_not_present = set(
            self.synset_dict.keys()).difference(synset_set)
        [self.synset_inv.pop(self.synset_dict[synset])
         for synset in synset_not_present]

        if len(synset_not_present) > 0:
            msg = (
                "The following categories are included in ShapeNetCore ver.%d's "
                "official mapping but not found in the dataset location %s: %s"
                ""
            ) % (version, data_dir, ", ".join(synset_not_present))
            warnings.warn(msg)

        # Extract model_id of each object from directory names.
        # Each grandchildren directory of data_dir contains an object, and the name
        # of the directory is the object's model_id.
        for synset in synset_set:
            self.synset_start_idxs[synset] = len(self.synset_ids)
            for model in os.listdir(path.join(data_dir, synset)):
                if not path.exists(path.join(data_dir, synset, model, self.model_dir)):
                    msg = (
                        "Object file not found in the model directory %s "
                        "under synset directory %s."
                    ) % (model, synset)
                    # warnings.warn(msg)
                    continue
                self.synset_ids.append(synset)
                self.model_ids.append(model)
            model_count = len(self.synset_ids) - self.synset_start_idxs[synset]
            self.synset_num_models[synset] = model_count
        self.model_ids, self.synset_ids = sort_jointly([self.model_ids, self.synset_ids], dim=0)
        self.classes = sorted(list(self.synset_inv.keys()))
        self.label_by_number = {k: v for v, k in enumerate(self.classes)}

        # adding train/val/test splits of the data 
        split_model_ids,split_synset_ids = [] , [] 
        for ii, model in enumerate(self.model_ids):
            found = splits[splits.modelId.isin([model])]["split"]
            if len(found) > 0:
                if found.item() in self.split:
                    split_model_ids.append(model)
                    split_synset_ids.append(self.synset_ids[ii])
        self.model_ids = split_model_ids
        self.synset_ids = split_synset_ids
        # self.model_ids = list(splits[splits.modelId.isin(self.model_ids)][splits.split.isin([self.split])]["modelId"])
        # self.synset_ids = list(splits[splits.modelId.isin(self.model_ids)][splits.split.isin([self.split])]["synsetId"])



    def __getitem__(self, idx: int) -> Dict:
        """
        Read a model by the given index.
        Args:
            idx: The idx of the model to be retrieved in the dataset.
        Returns:
            dictionary with following keys:
            - verts: FloatTensor of shape (V, 3).
            - faces: LongTensor of shape (F, 3) which indexes into the verts tensor.
            - synset_id (str): synset id
            - model_id (str): model id
            - label (str): synset label.
        """
        model = self._get_item_ids(idx)
        model_path = path.join(
            self.shapenet_dir, model["synset_id"], model["model_id"], self.model_dir
        )
        verts, faces, textures = self._load_mesh(model_path)
        label_str = self.synset_dict[model["synset_id"]]
        # model["verts"] = verts
        # model["faces"] = faces
        # model["textures"] = textures
        # model["label"] = self.synset_dict[model["synset_id"]]
        verts = torch_center_and_normalize(verts.to(torch.float), p=self.dset_norm)

        verts_rgb = torch.ones_like(verts)[None]  # (1, V, 3)
        textures = Textures(verts_rgb=verts_rgb)
        mesh = Meshes(
            verts=[verts],
            faces=[faces],
            textures=textures
        )
        points = trimesh.Trimesh(vertices=verts.numpy(), faces=faces.numpy()).sample(self.nb_points, False)
        points = torch.from_numpy(points).to(torch.float)
        points = torch_center_and_normalize(points, p=self.dset_norm)
        return self.label_by_number[label_str], mesh, points, 1.0
        # return model


class ScanObjectNN(torch.utils.data.Dataset):
    """
    This class loads ScanObjectNN from a given directory into a Dataset object.
    ScanObjjectNN is a point cloud dataset of realistic shapes of from the ScanNet dataset and can be downloaded from
    https://github.com/hkust-vgd/scanobjectnn .
    """

    def __init__(
        self,
        data_dir,
        split,
        nb_points,
        normals: bool = False,
        suncg: bool = False,
        variant: str = "obj_only",
        dset_norm: str = "inf",

    ):
        """
        Store each object's synset id and models id from data_dir.
        Args:
            data_dir: Path to ShapeNetCore data.
            synsets: List of synset categories to load from ShapeNetCore in the form of
                synset offsets or labels. A combination of both is also accepted.
                When no category is specified, all categories in data_dir are loaded.
            version: (int) version of ShapeNetCore data in data_dir, 1 or 2.
                Default is set to be 1. Version 1 has 57 categories and verions 2 has 55
                categories.
                Note: version 1 has two categories 02858304(boat) and 02992529(cellphone)
                that are hyponyms of categories 04530566(watercraft) and 04401088(telephone)
                respectively. You can combine the categories manually if needed.
                Version 2 doesn't have 02858304(boat) or 02834778(bicycle) compared to
                version 1.
            load_textures: Boolean indicating whether textures should loaded for the model.
                Textures will be of type TexturesAtlas i.e. a texture map per face.
            texture_resolution: Int specifying the resolution of the texture map per face
                created using the textures in the obj file. A
                (texture_resolution, texture_resolution, 3) map is created per face.
        """
        super().__init__()
        self.data_dir = data_dir
        self.nb_points = nb_points
        self.normals = normals
        self.suncg = suncg
        self.variant = variant
        self.dset_norm = dset_norm
        self.split = split
        self.classes = {0: 'bag', 10: 'bed', 1: 'bin', 2: 'box', 3: 'cabinet', 4: 'chair', 5: 'desk', 6: 'display',
                        7: 'door', 11: 'pillow', 8: 'shelf', 12: 'sink', 13: 'sofa', 9: 'table', 14: 'toilet'}

        self.labels_dict = {"train": {}, "test": {}}
        self.objects_paths = {"train": [], "test": []}

        if self.variant != "hardest":
            pcdataset = pd.read_csv(os.path.join(
                data_dir, "split_new.txt"), sep="\t", names=['obj_id', 'label', "split"])
            for ii in range(len(pcdataset)):
                if pcdataset["split"][ii] != "t":
                    self.labels_dict["train"][pcdataset["obj_id"]
                                            [ii]] = pcdataset["label"][ii]
                else:
                    self.labels_dict["test"][pcdataset["obj_id"]
                                            [ii]] = pcdataset["label"][ii]

            all_obj_ids = glob.glob(os.path.join(self.data_dir, "*/*.bin"))
            filtered_ids = list(filter(lambda x: "part" not in os.path.split(
                x)[-1] and "indices" not in os.path.split(x)[-1], all_obj_ids))

            self.objects_paths["train"] = sorted(
                [x for x in filtered_ids if os.path.split(x)[-1] in self.labels_dict["train"].keys()])
            self.objects_paths["test"] = sorted(
                [x for x in filtered_ids if os.path.split(x)[-1] in self.labels_dict["test"].keys()])
        else:
            filename = os.path.join(data_dir, "{}_objectdataset_augmentedrot_scale75.h5".format(self.split))
            with h5py.File(filename, "r") as f:
                self.labels_dict[self.split] = np.array(f["label"])
                self.objects_paths[self.split] = np.array(f["data"])
            #     print("1############", len(self.labels_dict[self.split]))
            # print("2############", len(self.labels_dict[self.split]))


    def __getitem__(self, idx: int) -> Dict:
        """
        Read a model by the given index. no mesh is availble in this dataset so retrun None and correction factor of 1.0

        """
        if self.variant != "hardest":
            obj_path = self.objects_paths[self.split][idx]
            # obj_path,label
            points = self.load_pc_file(obj_path)
            # sample the required number of points randomly
            points = points[np.random.randint(points.shape[0], size=self.nb_points),:]
            # print(pc.min(),classes[label],obj_path)
            label = self.labels_dict[self.split][os.path.split(obj_path)[-1]]
        else:

            points = self.objects_paths[self.split][idx]
            label = self.labels_dict[self.split][idx]

        points = torch.from_numpy(points).to(torch.float)
        points = torch_center_and_normalize(points, p=self.dset_norm)
        return label, None, points, 1.0

    def __len__(self):
        return len(self.objects_paths[self.split])

    def load_pc_file(self,filename):
        #load bin file
        # pc=np.fromfile(filename, dtype=np.float32)
        pc = np.fromfile(filename, dtype=np.float32)

        #first entry is the number of points
        #then x, y, z, nx, ny, nz, r, g, b, label, nyu_label
        if(self.suncg):
            pc = pc[1:].reshape((-1, 3))
        else:
            pc = pc[1:].reshape((-1, 11))

        #only use x, y, z for now
        if self.variant == "with_bg":
            pc = np.array(pc[:, 0:3])
            return pc

        else:
            ##To remove backgorund points
            ##filter unwanted class
            filtered_idx = np.intersect1d(np.intersect1d(np.where(
                pc[:, -1] != 0)[0], np.where(pc[:, -1] != 1)[0]), np.where(pc[:, -1] != 2)[0])
            (values, counts) = np.unique(pc[filtered_idx, -1], return_counts=True)
            max_ind = np.argmax(counts)
            idx = np.where(pc[:, -1] == values[max_ind])[0]
            pc = np.array(pc[idx, 0:3])
            return pc


def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc


class PartNormalDataset(torch.utils.data.Dataset):
    def __init__(self, root='./data/shapenetcore_partanno_segmentation_benchmark_v0_normal', npoints=2500, split='train', class_choice=None, normal_channel=False):
        self.npoints = npoints
        self.root = root
        self.catfile = os.path.join(self.root, 'synsetoffset2category.txt')
        self.cat = {}
        self.normal_channel = normal_channel

        with open(self.catfile, 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.cat[ls[0]] = ls[1]
        self.cat = {k: v for k, v in self.cat.items()}
        self.classes_original = dict(zip(self.cat, range(len(self.cat))))

        if not class_choice is None:
            self.cat = {k: v for k, v in self.cat.items() if k in class_choice}
        # print(self.cat)

        self.meta = {}
        with open(os.path.join(self.root, 'train_test_split', 'shuffled_train_file_list.json'), 'r') as f:
            train_ids = set([str(d.split('/')[2]) for d in json.load(f)])
        with open(os.path.join(self.root, 'train_test_split', 'shuffled_val_file_list.json'), 'r') as f:
            val_ids = set([str(d.split('/')[2]) for d in json.load(f)])
        with open(os.path.join(self.root, 'train_test_split', 'shuffled_test_file_list.json'), 'r') as f:
            test_ids = set([str(d.split('/')[2]) for d in json.load(f)])
        for item in self.cat:
            # print('category', item)
            self.meta[item] = []
            dir_point = os.path.join(self.root, self.cat[item])
            fns = sorted(os.listdir(dir_point))
            # print(fns[0][0:-4])
            if split == 'trainval':
                fns = [fn for fn in fns if (
                    (fn[0:-4] in train_ids) or (fn[0:-4] in val_ids))]
            elif split == 'train':
                fns = [fn for fn in fns if fn[0:-4] in train_ids]
            elif split == 'val':
                fns = [fn for fn in fns if fn[0:-4] in val_ids]
            elif split == 'test':
                fns = [fn for fn in fns if fn[0:-4] in test_ids]
            else:
                print('Unknown split: %s. Exiting..' % (split))
                exit(-1)

            # print(os.path.basename(fns))
            for fn in fns:
                token = (os.path.splitext(os.path.basename(fn))[0])
                self.meta[item].append(os.path.join(dir_point, token + '.txt'))

        self.datapath = []
        for item in self.cat:
            for fn in self.meta[item]:
                self.datapath.append((item, fn))

        self.classes = {}
        for i in self.cat.keys():
            self.classes[i] = self.classes_original[i]

        # Mapping from category ('Chair') to a list of int [10,11,12,13] as segmentation labels
        self.seg_classes = {'Earphone': [16, 17, 18], 'Motorbike': [30, 31, 32, 33, 34, 35], 'Rocket': [41, 42, 43],
                            'Car': [8, 9, 10, 11], 'Laptop': [28, 29], 'Cap': [6, 7], 'Skateboard': [44, 45, 46],
                            'Mug': [36, 37], 'Guitar': [19, 20, 21], 'Bag': [4, 5], 'Lamp': [24, 25, 26, 27],
                            'Table': [47, 48, 49], 'Airplane': [0, 1, 2, 3], 'Pistol': [38, 39, 40],
                            'Chair': [12, 13, 14, 15], 'Knife': [22, 23]}
        self.clsnb_to_label = {ii: k for ii,k in enumerate(sorted(list(self.seg_classes.keys())))}
        self.cls_to_parts = {ii: self.seg_classes[self.clsnb_to_label[ii]] for ii in range(len(self.seg_classes))}

        self.part_classes = []
        for k, v in self.classes.items():
            self.part_classes.extend(self.seg_classes[k])
        self.part_classes = sorted(self.part_classes)       
        skeys = sorted(list(self.seg_classes.keys()))
        self.parts_per_class = [len(self.seg_classes[x]) for x in skeys]
        self.cache = {}  # from index to (point_set, cls, seg) tuple
        self.cache_size = 20000

    def __getitem__(self, index):
        if index in self.cache:
            point_set, cls, seg = self.cache[index]
        else:
            fn = self.datapath[index]
            cat = self.datapath[index][0]
            cls = self.classes[cat]
            cls = np.array([cls]).astype(np.int32)
            data = np.loadtxt(fn[1]).astype(np.float32)
            if not self.normal_channel:
                point_set = data[:, 0:3]
            else:
                point_set = data[:, 0:6]
            seg = data[:, -1].astype(np.int32)
            if len(self.cache) < self.cache_size:
                self.cache[index] = (point_set, cls, seg)
        point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])
        
        
        true_nb_points = point_set.shape[0]
        # resample
        # choice = np.random.choice(len(seg), self.npoints, replace=True)
        if true_nb_points >= self.npoints:
            choice = np.arange(self.npoints)
            real_points_mask = np.ones(self.npoints, dtype=np.int)
        else :
            choice = np.ones(self.npoints,dtype=np.int) * (true_nb_points-1)
            choice[:true_nb_points] = np.arange(true_nb_points)
            real_points_mask = np.zeros(self.npoints, dtype=np.int)
            real_points_mask[:true_nb_points] = 1


        point_set = point_set[choice, :]
        seg = seg[choice]

        return point_set, cls, seg, self.cls_to_parts[int(cls)][0], len(self.cls_to_parts[int(cls)]), real_points_mask

    def __len__(self):
        return len(self.datapath)
