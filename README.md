# MVTN: Multi-View Transformation Network for 3D Shape Recognition (ICCV 2021)
By [Abdullah Hamdi](https://abdullahamdi.com/), [Silvio Giancola](https://www.silviogiancola.com/), [Bernard Ghanem](http://www.bernardghanem.com/)
### [paper](https://arxiv.org/pdf/2011.13244.pdf) | Video | Tutorial . <br>

<br>

<img src="https://github.com/ajhamdi/MVTN/blob/master/doc/pipeline.png" width="80%" alt="MVTN pipeline" align=center>

The official Pytroch code of ICCV 2021 paper [MVTN: Multi-View Transformation Network for 3D Shape Recognition](https://arxiv.org/abs/2011.13244) . MVTN learns to transform the rendering parameters of a 3D object to improve the perspectives for better recognition by multi-view netowkrs. Without extra supervision or add loss, MVTN improve the performance in 3D classification and shape retrieval. MVTN achieves state-of-the-art performance on ModelNet40, ShapeNet Core55, and the most recent and realistic ScanObjectNN dataset (up to 6% improvement).  

## Citation
If you find our work useful in your research, please consider citing:
```bibtex
@inproceedings{hamdi2021mvtn,
  title={MVTN: Multi-View Transformation Network for 3D Shape Recognition},
  author={Hamdi, Abdullah and Giancola, Silvio and Ghanem, Bernard},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  year={2021}
}
```



## Requirement
This code is tested with Python 3.7 and Pytorch >= 1.4

- install Pytorch3d from [here](https://github.com/facebookresearch/pytorch3d/blob/master/INSTALL.md)

- install other helper libraries 

```bash
conda install pandas
pip install imageio
conda install -c conda-forge trimesh
pip install einops
``` 

## Usage

The main Python script in the root directorty `run_mvtn.py`. 

First download the datasets and unzip inside the `data/` directories as follows: 

- ModelNet40 [this link](https://drive.google.com/file/d/157W0qYR2yQAc5qKmXlZuHms66wmUM8Hi/view?usp=sharing) (ModelNet objects meshes are simplified to fit the GPU and allows for backpropogation ). 

- ShapeNet Core55 v2 [this link](https://shapenet.org/download/shapenetcore) ( You need to create an account) 

- ScanObjectNN [this link](https://drive.google.com/file/d/15xhYA8SC5EdLKZA_xV0FXyRy8f-qGMs5/view?usp=sharing) (ScanObjectNN with its three main variants [`obj_only` ,`with_bg` , `hardest`] controlled by the `--dset_variant` option  ). 

Then you can run MVTN with 
```bash
python run_mvtn.py --data_dir data/ModelNet40/ --run_mode train --mvnetwork mvcnn --nb_views 8 --views_config learned_spherical  
```
- `--data_dir` the data directory. The dataloader is picked adaptively from `custom_dataset.py` based on the choice between "ModelNet40", "ShapeNetCore.v2", or the "ScanObjectNN" choice.
- `--run_mode` is the run mode. choices: "train"(train for classification), "test_cls"(test classification after training), "test_retr"(test retrieval after training), "test_rot"(test rotation robustness after training), "test_occ"(test occlusion robustness after training)
- `--mvnetwork` is the multi-view network used in the pipeline. Choices: "[mvcnn](https://github.com/RBirkeland/MVCNN-PyTorch)" , "[rotnet](https://github.com/kanezaki/pytorch-rotationnet)", "[viewgcn](https://github.com/weixmath/view-GCN)"
- `--views_config` is one of six view selection methods that are either learned or heuristics :  choices: "circular", "random", "spherical" "learned_circular" , "learned_spherical" , "learned_direct". Only the ones that are learned are MVTN variants.
- `--resume` continue training from last checkpoint.
- `--pc_rendering` : a Flag if you want to use point clouds instead of mesh data and point cloud rendering instead of mesh rendering. This should be default when only point cloud data is available ( like in ScanObjectNN dataset)
- `--object_color`: is the uniform color of the mesh or object rendered. default="white", choices=["white", "random", "black", "red", "green", "blue", "custom"]
Other parameters can be founded in `config.yaml` configuration file or run `python run_mvtn.py -h`. The default parameters are the ones used in the paper.

The results will be saved in `results/00/0001/` folder that contaions the camera view points and the renderings of some example as well the checkpoints and the logs.
<br>


## Other files
- `models/renderer.py` contains the main Pytorch3D  differentiable renderer class that can render multi-view images for point clouds and meshes adaptively.
- `models/mvtn.py` contains a standalone class for MVTN that can be used with any other pipeline.
- `custom_dataset.py` includes all the pytorch dataloaders for 3D datasets: ModelNEt40, SahpeNet core55 ,ScanObjectNN, and ShapeNet Parts 
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
