# -*- coding: utf-8 -*-
"""
@author: Anton Wang
"""

import argparse
import numpy as np

parser=argparse.ArgumentParser(description='Mesh simplify')
parser.add_argument('-i', type=str, default=None, help='Please provide the input file path of an existing 3d model.')
parser.add_argument('-o', type=str, default=None, help='Please provide the output file path of the simplified model.')
parser.add_argument('-r', type=np.float, default=0.5, help='Simplification ratio (0<r<=1).')
parser.add_argument('-t', type=np.float, default=0, help='Threshold parameter for valid pair selection (>=0).')
args=parser.parse_args()

input_filepath=args.i
output_filepath=args.o
threshold=args.t
simplify_ratio=args.r

from models.class_mesh_simplify import mesh_simplify

# Here, point and vertex are same terms
# Read 3d model, initialization (points/vertices, faces, edges), compute the Q matrices for all the initial vertices
model=mesh_simplify(input_filepath, threshold, simplify_ratio)

# Select all valid pairs.
model.generate_valid_pairs()

# Compute the optimal contraction target v_opt for each valid pair (v1, v2)
# The error v_opt.T*(Q1+Q2)*v_opt of this target vertex becomes the cost of contracting that pair.
# Place all the pairs in a heap keyed on cost with the minimum cost pair at the top
model.calculate_optimal_contraction_pairs_and_cost()

# Iteratively remove the pair (v1, v2) of least cost from the heap
# contract this pair, and update the costs of all valid pairs involving (v1, v2).
# until existing points = ratio * original points
model.iteratively_remove_least_cost_valid_pairs()

# Generate the simplified 3d model (points/vertices, faces)
model.generate_new_3d_model()

# Output the model to output_filepath
model.output(output_filepath)