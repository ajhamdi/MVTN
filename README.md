# MVTN: Multi-View Transformation Network for 3D Shape Recognition (ICCV 2021)
By [Abdullah Hamdi](https://abdullahamdi.com/), [Silvio Giancola](https://www.silviogiancola.com/), [Bernard Ghanem](http://www.bernardghanem.com/)
### [Paper](https://arxiv.org/pdf/2011.13244.pdf) | [Video](https://youtu.be/1zaHx8ztlhk) | Tutorial . <br>
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/mvtn-multi-view-transformation-network-for-3d/3d-point-cloud-classification-on-scanobjectnn)](https://paperswithcode.com/sota/3d-point-cloud-classification-on-scanobjectnn?p=mvtn-multi-view-transformation-network-for-3d)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/mvtn-multi-view-transformation-network-for-3d/3d-object-retrieval-on-modelnet40)](https://paperswithcode.com/sota/3d-object-retrieval-on-modelnet40?p=mvtn-multi-view-transformation-network-for-3d)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/mvtn-multi-view-transformation-network-for-3d/3d-object-retrieval-on-shapenetcore-55)](https://paperswithcode.com/sota/3d-object-retrieval-on-shapenetcore-55?p=mvtn-multi-view-transformation-network-for-3d)[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/mvtn-multi-view-transformation-network-for-3d/3d-point-cloud-classification-on-modelnet40)](https://paperswithcode.com/sota/3d-point-cloud-classification-on-modelnet40?p=mvtn-multi-view-transformation-network-for-3d)
<br>

<img src="https://github.com/ajhamdi/MVTN/blob/master/doc/pipeline.png" width="80%" alt="MVTN pipeline" align=center>

The official Pytroch code of ICCV 2021 paper [MVTN: Multi-View Transformation Network for 3D Shape Recognition](https://arxiv.org/abs/2011.13244). MVTN learns to transform the rendering parameters of a 3D object to improve the perspectives for better recognition by multi-view netowkrs. Without extra supervision or add loss, MVTN improve the performance in 3D classification and shape retrieval. MVTN achieves state-of-the-art performance on ModelNet40, ShapeNet Core55, and the most recent and realistic ScanObjectNN dataset (up to 6% improvement).  

## Citation
If you find our work useful in your research, please consider citing:
```bibtex
@InProceedings{Hamdi_2021_ICCV,
    author    = {Hamdi, Abdullah and Giancola, Silvio and Ghanem, Bernard},
    title     = {MVTN: Multi-View Transformation Network for 3D Shape Recognition},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2021},
    pages     = {1-11}
}
```



## Requirement
This code is tested with Python 3.7 and Pytorch >= 1.5

- install [Pytorch3d](https://github.com/facebookresearch/pytorch3d/blob/master/INSTALL.md) as follows
```bash
conda create -y -n MVTN python=3.7
conda activate MVTN
conda install -c pytorch pytorch=1.7.1 torchvision cudatoolkit=10.2
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
conda install -c bottler nvidiacub
conda install pytorch3d -c pytorch3d
``` 
- install other helper libraries 

```bash
conda install pandas
conda install -c conda-forge trimesh
pip install einops imageio scipy matplotlib tensorboard h5py metric-learn
``` 

## Usage: 3D Classification & Retrieval

The main Python script is in the root directory `run_mvtn.py`. 

First, download the datasets and unzip them inside the `data/` directories as follows: 

- ModelNet40 [this link](https://drive.google.com/file/d/157W0qYR2yQAc5qKmXlZuHms66wmUM8Hi/view?usp=sharing) (ModelNet objects meshes are simplified to fit the GPU and allow for backpropogation ). 

- ShapeNet Core55 v2 [this link](https://shapenet.org/download/shapenetcore) ( You need to create an account) . Make sure to put the two files [shapenet_synset_dict_v2.json](https://github.com/ajhamdi/MVTN/blob/master/data/shapenet_synset_dict_v2.json) and [shapenet_split.csv](https://github.com/ajhamdi/MVTN/blob/master/data/shapenet_split.csv) inside the ShapeNet directory. 

- ScanObjectNN [this link](https://drive.google.com/file/d/15xhYA8SC5EdLKZA_xV0FXyRy8f-qGMs5/view?usp=sharing) (ScanObjectNN with its three main variants [`obj_only` ,`with_bg` , `hardest`] controlled by the `--dset_variant` option  ). 

Then you can run MVTN with 
```bash
python run_mvtn.py --data_dir data/ModelNet40/ --run_mode train --mvnetwork mvcnn --nb_views 8 --views_config learned_spherical  
```
- `--data_dir` the data directory. The dataloader is picked adaptively from `custom_dataset.py` based on the choice between "ModelNet40", "ShapeNetCore.v2", or the "ScanObjectNN" choice.
- `--run_mode` is the run mode. choices: "train"(train for classification), "test_cls"(test classification after training), "test_retr"(test retrieval after training), "test_rot"(test rotation robustness after training), "test_occ"(test occlusion robustness after training)
- `--mvnetwork` is the multi-view network used in the pipeline. Choices: "[mvcnn](https://github.com/RBirkeland/MVCNN-PyTorch)" , "[rotnet](https://github.com/kanezaki/pytorch-rotationnet)", "[viewgcn](https://github.com/weixmath/view-GCN)"
- `--views_config` is one of six view selection methods that are either learned or heuristics :  choices: "circular", "random", "spherical" "learned_circular" , "learned_spherical" , "learned_direct". Only the ones that are learned are MVTN variants.
- `--resume` a flag to continue training from last checkpoint.
- `--pc_rendering` : a flag if you want to use point clouds instead of mesh data and point cloud rendering instead of mesh rendering. This should be default when only point cloud data is available ( like in ScanObjectNN dataset)
- `--object_color`: is the uniform color of the mesh or object rendered. default="white", choices=["white", "random", "black", "red", "green", "blue", "custom"]

Other parameters can be founded in `config.yaml` configuration file or run `python run_mvtn.py -h`. The default parameters are the ones used in the paper.

The results will be saved in `results/00/0001/` folder that contaions the camera view points and the renderings of some example as well the checkpoints and the logs.

**Note**: For best performance on point cloud tasks, please set `canonical_distance : 1.0` in the `config.yaml` file. For mesh tasks, keep as is. 
<br>


## Other files
- `models/renderer.py` contains the main Pytorch3D  differentiable renderer class that can render multi-view images for point clouds and meshes adaptively.
- `models/mvtn.py` contains a standalone class for MVTN that can be used with any other pipeline.
- `custom_dataset.py` includes all the pytorch dataloaders for 3D datasets: ModelNet40, SahpeNet core55 ,ScanObjectNN, and ShapeNet Parts 
- `blender_simplify.py` is the Blender code used to simplify the meshes with `simplify_mesh` function from `util.py` as the   following :  
```python 
simplify_ratio  = 0.05 # the ratio of faces to be maintained after simplification 
input_mesh_file = os.path.join(data_dir,"ModelNet40/plant/train/plant_0014.off") 
mymesh, reduced_mesh = simplify_mesh(input_mesh_file,simplify_ratio=simplify_ratio)
```
The output simplified mesh will be saved in the same directory of the original mesh with "SMPLER" appended to the name

## Misc
- Please open an issue or contact Abdullah Hamdi (abdullah.hamdi@kaust.edu.sa) if there is any question.

## Acknoledgements
This paper and repo borrows codes and ideas from several great github repos:
[MVCNN pytorch](https://github.com/RBirkeland/MVCNN-PyTorch) , [view GCN](https://github.com/weixmath/view-GCN), [RotationNet](https://github.com/kanezaki/pytorch-rotationnet) and most importantly the great [Pytorch3D](https://github.com/facebookresearch/pytorch3d) library.
## License
The code is released under MIT License (see LICENSE file for details).
