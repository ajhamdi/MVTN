import bpy
import sys
import time
import argparse

'''
Description: A Python Tool that decimates an OBJ 3D model into lower resolutions (in nb of faces)
It uses the Blender Python API.
Requirements: You need only to install Blender first on the OS in question
          Example in Ubuntu Server 16.04: 'sudo apt-get install blender'
          Example in Fedora 26:           'sudo dnf install blender'
          Make sure you can call Blender from cmd/terminal etc...
Usage: blender -b -P blenderSimplify.py -- --ratio 0.5 --inm 'Original_Mesh.obj' --outm 'Output_Mesh.obj'
After --inm:  you specify the original mesh to import for decimation
      --outm: you specify the final output mesh name to export
      --ratio: this ratio should be between 0.1 and 1.0(no decimation occurs). If you choose
      Per example --ratio 0.5 meaning you half the number of faces so if your model is 300K faces
      it will be exported as 150K faces
PS: this tool does not try to preserve the integrity of the mesh so be carefull in choosing
the ratio (try not choose a very low ratio)
Enjoy!
'''
 
def get_args():
  parser = argparse.ArgumentParser()
 
  # get all script args
  _, all_arguments = parser.parse_known_args()
  double_dash_index = all_arguments.index('--')
  script_args = all_arguments[double_dash_index + 1: ]
 
  # add parser rules
  parser.add_argument('-r', '--ratio', help="Ratio of reduction, Example: 0.5 mean half number of faces ")
  parser.add_argument('-in', '--inm', help="Original Model")
  parser.add_argument('-out', '--outm', help="Decimated output file")
  parsed_script_args, _ = parser.parse_known_args(script_args)
  return parsed_script_args
 
args = get_args()
decimateRatio = float(args.ratio)
print(decimateRatio)

input_model = str(args.inm)
print(input_model)

output_model = str(args.outm)
print(output_model)

print('\n Clearing blender scene (default garbage...)')
# deselect all
bpy.ops.object.select_all(action='DESELECT')

# selection
bpy.data.objects['Camera'].select_set(True)

# remove it
bpy.ops.object.delete() 

# Clear Blender scene
# select objects by type
for o in bpy.data.objects:
    if o.type == 'MESH':
        o.select_set(True)
    else:
        o.select_set(False)

# call the operator once
bpy.ops.object.delete()

print('\n Beginning the process of Decimation using Blender Python API ...')
bpy.ops.import_scene.obj(filepath=input_model)
print('\n Obj file imported successfully ...')
modifierName='DecimateMod'

print('\n Creating and object list and adding meshes to it ...')
objectList=bpy.data.objects
meshes = []
for obj in objectList:
  if(obj.type == "MESH"):
    meshes.append(obj)

print("{} meshes".format(len(meshes)))

for i, obj in enumerate(meshes):
  bpy.context.scene.objects.active = obj
  print("{}/{} meshes, name: {}".format(i, len(meshes), obj.name))
  print("{} has {} verts, {} edges, {} polys".format(obj.name, len(obj.data.vertices), len(obj.data.edges), len(obj.data.polygons)))
  modifier = obj.modifiers.new(modifierName,'DECIMATE')
  modifier.ratio = decimateRatio
  modifier.use_collapse_triangulate = True
  bpy.ops.object.modifier_apply(apply_as='DATA', modifier=modifierName)
  print("{} has {} verts, {} edges, {} polys after decimation".format(obj.name, len(obj.data.vertices), len(obj.data.edges), len(obj.data.polygons)))

bpy.ops.export_scene.obj(filepath=output_model)
print('\n Process of Decimation Finished ...')
