# generate train_lists.txt, val_lists.txt, test_lists.txt
# each including lines of the lists (the ones in list/) of views
import numpy as np
from glob import glob
import os.path as p
import re
import shutil
import os

LISTS = '/tmp3/weitang114/ibsr/data_modelnet40/view/list/'
TRAIN = LISTS + 'train/'
TEST = LISTS + 'test/'
CLASSES = '../classes.txt'

TRAIN_OUT = './train_lists.txt'
TEST_OUT = './test_lists.txt'
VAL_OUT = './val_lists.txt'


def out_lists(outfile, lists, class_index):
    with open(outfile, 'a+') as f:
        for l in lists:
            print>>f, '%s %d' % (l, class_index)

def clear_outfiles():
    try:
        os.remove(TRAIN_OUT)
    except:
        pass
    try:
        os.remove(VAL_OUT)
    except:
        pass
    try:
        os.remove(TEST_OUT)
    except:
        pass

clear_outfiles()

classes = np.loadtxt(CLASSES, dtype=str)
for c_index, c in enumerate(classes):
    lists = glob(p.join(TRAIN, c, '*.txt'))
    def get_id_of_list(l):
        try:
            id_ = int(re.split('_|\.', p.basename(l))[-3])
            return id_
        except:
            print l

    lists = sorted(lists, key=get_id_of_list)
    
    # train/val
    train_lists = lists[:-10]
    val_lists = lists[-10:]

    out_lists(TRAIN_OUT, train_lists, c_index)
    out_lists(VAL_OUT, val_lists, c_index)

    
    test_lists = glob(p.join(TEST, c, '*.txt'))
    test_lists = sorted(test_lists, key=get_id_of_list)
    out_lists(TEST_OUT, test_lists, c_index)
    



