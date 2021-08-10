import numpy as np
import os
import os.path as p
from glob import glob
import errno


CLASSES = '../classes.txt'
LISTS_TRAIN = './list/train/'
LISTS_TEST =  './list/test/'
IMAGES = '/tmp3/weitang114/ibsr/data_modelnet40/view/classes/'

classes = np.loadtxt(CLASSES, dtype=str)

def mkdir_p(path):
    try:
        os.mkdir(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def gen_list(listdir, imagedir, clz):
    models = os.listdir(imagedir)
    for m in models: 
        imfiles = glob(p.join(imagedir, m, '*.png'))
        imfiles = sorted(imfiles, key=lambda s:int(s.split('.')[-2]))
        listfile = p.join(listdir, '%s.txt' % m)

        with open(listfile, 'w+') as f:
            print>>f, clz # label
            print>>f, 12 # 12 views
            for imfile in imfiles:
                print>>f, imfile


for cind, c in enumerate(classes):
    listdir_train = LISTS_TRAIN + c
    listdir_test = LISTS_TEST + c
    imagedir_train = IMAGES + c + '/train/'
    imagedir_test = IMAGES + c + '/test/'

    mkdir_p(LISTS_TRAIN + c)
    mkdir_p(LISTS_TEST + c)

    gen_list(listdir_train, imagedir_train, cind)
    gen_list(listdir_test, imagedir_test, cind)

