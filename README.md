# MVT: Multi-View Transformer for 3D Classification and Segmentation
By [Abdullah Hamdi](https://abdullahamdi.com/), Silvio Giancola, [Bernard Ghanem](http://www.bernardghanem.com/)
### [paper | Video | Tutorial . <br>
I was working on a multiview transformer: a network that can predict how to look properly at a 3d object in MVCNN setup to get max classification/segmentation
this came after earlier unsuccessful attempts on multiview for point cloud DGCNN (edited).
this follows after that recent paper from David ross showed that multiview can match other 3d deep learning techniques if done properly
basically Multi-view is the best way to understand 3d ( at least it is SOTA in classification, segmentation now )
most previous methods assumed a fixed/random/heuristic view points for the MV methods
I am challenging this by LEARNING the view angles that minimize the downstream task loss ( ie classification now  and then part segmentation and scene segementation )
to do that I am using recent Pytorch3D differentiable renderer in order to backpropagate through the rendering process

<img src="https://github.com/ajhamdi/MVTN/master/doc/pipeline.png" width="80%" alt="attack pipeline" align=center>



## Requirement
This code is tested with Python 3.7 and Pytorch >= 1.4

- install Pytorch3d from [here](https://github.com/facebookresearch/pytorch3d/blob/master/INSTALL.md)

- install other helper libraries 

```bash
conda install pandas
pip install imageio
conda install -c conda-forge trimesh
``` 

## Usage
There are two main Python scripts in the root directorty: 
- `mvt_cls.py` -- MVT for ModelNet40 classification (**Working in progress**)
- `mvt_partseg.py` -- MVT code for segmentation ( **TODO** )  

First download the data folder from [this link](https://drive.google.com/drive/folders/1CcaD2zWfRPYom05Goi4PTlpt-VHkpYph?usp=sharing) and unzip its content (ModelNet is simplified to fir the GPU ). Then you can run MVT with 
```
python mvt_cls.py --epochs 100 --nb_views 4 --batch-size 4 --selection_type canonical --image_data data/view/classes/ --mesh_data data/ModelNet40/ --simplified_mesh --pretrained --resume checkpoint/resnet18_checkpoint.pt --learning-rate 0.0001
```
- `selection_type` is one of three view selections :  choices=("canonical", "random", "learned")
- `resume` continue training from this checkpoint.
- `pretrained` use ImageNet pretrained networks
- `image_data` the folder for prerendered images ( not needed really ).
- `mesh_data` : the folder that has the meshes ( ModelNEt40 ) 
- `simplified_mesh` : a Flag if you want to use the simplified meshes not the full meshes 

Other parameters can be founded in the script, or run `python mvt_cls.py -h`. The default parameters are the ones used in the paper.

The results will be saved in `cameras/` folder for the camera view points and the renderings of some example are saved in `renderings/`.
<br>
The `ViewSelector` object in `ops.py` is the main class in this work .. it has options "random" , "canonical" and "learned" ... **TODO** need to be updated with more setup

## Other files
- `checkpoint/` folder saves the checkpoints for trinmed models 
- `blender_simplify.py` is the code used to simplify the meshes with following function.. please run while you are insied the project directory 
```python 
def simplify_mesh(input_file,simplify_ratio=0.05):
    """
    a function to reduce the poly of meshe `input_file` by some ratio `simplify_ratio`
    Reuturns : the mesh in `input_file` as Trimesh object and the simplified mesh as Trimehs object and saves the simplified mesh with the 
    """
    if input_file[-3::] == "off":
        input_obj_file = input_file.replace(".off",".obj")
        input_off_file = input_file
    elif input_file[-3::] == "obj":
        input_off_file = input_file.replace(".obj",".off")
        input_obj_file = input_file
    mymesh = trimesh.load(input_file)
    input_file = input_file[:-4]
    output_obj_file = "{}_SMPL.obj".format(input_file) 
    if not os.path.isfile(input_obj_file):
        _ = mymesh.export(input_obj_file)
    command = "blender -b -P {} -- --ratio {} --inm '{}' --outm '{}'".format(os.path.join(project_dir,"blender_simplify.py"),simplify_ratio,input_obj_file,output_obj_file)
    os.system(command)
    reduced_mesh = trimesh.load(output_obj_file)    
    return mymesh ,  reduced_mesh

mymesh, reduced_mesh = simplify_mesh(input_off_file,simplify_ratio=0.05)
```

## Misc
- 

## Acknoledgements
This paper and repo borrows codes and ideas from several great github repos:
[MVCNN pytorch](https://github.com/RBirkeland/MVCNN-PyTorch) , [view GCN](https://github.com/weixmath/view-GCN).
## License
The code is released under MIT License (see LICENSE file for details).
