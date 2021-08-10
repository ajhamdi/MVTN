#!/bin/bash

# script for downloading multi-view images generated in [Su et al. 2015]
# [Su et al. 2015] H. Su, S. Maji, E. Kalogerakis, E. Learned-Miller. Multi-view Convolutional Neural Networks for 3D Shape Recognition. ICCV2015.

# download tar files
wget http://maxwell.cs.umass.edu/mvcnn-data/modelnet40v1png.tar
wget http://maxwell.cs.umass.edu/mvcnn-data/modelnet40v2png.tar

# extract tar files
tar xvf modelnet40v1png.tar
tar xvf modelnet40v2png.tar
