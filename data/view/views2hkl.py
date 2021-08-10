import hickle as hkl
import os.path as p
import numpy as np
from PIL import Image
from glob import glob
import re
import cv2

IMAGE_ROOT = './classes/'
HKL_DIR = './hkl/'

TRAIN_HKL = p.join(HKL_DIR, 'train.hkl')
TEST_HKL = p.join(HKL_DIR, 'test.hkl')
CLASSES = '/tmp3/weitang114/ibsr/data_modelnet40/classes.txt'
W = 256
H = 256

with open(CLASSES) as f: 
    classes = [l.strip() for l in f.readlines()]


def do(train_test):
    images = []
    view_directions = []
    labels = []
    for i,class_ in enumerate(classes):
        print i
        d = p.join(IMAGE_ROOT, class_, train_test)
        
        subdirs = glob(p.join(d, '*'))
        subdirs = sorted(subdirs, key=lambda d: int(re.split('_|\.', p.basename(d))[-2])) # airplane_0103.off
        for j,subd in enumerate(subdirs):
            id_ = p.basename(subd).split('.')[0] # airplane_0102

            for k in range(12):
                imagefile = p.join(subd, '%s.%d.png' % (id_, k))

                with Image.open(imagefile) as img:
                    img = img.resize((W, H))
                    img = img.convert('RGB') # 1 channel to 3 channels
                    arr = np.array(img)

                    images.append(arr)
                    labels.append(i)
                    view_directions.append(k)

    
    images = np.array(images)
    labels = np.array(labels)
    view_directions = np.array(view_directions)
    print images.shape, labels.shape
    print images.dtype, labels.dtype
    
    return images, labels, view_directions

#train
images, labels, views = do('train')
data = {
        'x': images,
        'y': labels,
        'view': views,
        'classes': classes
}
hkl.dump(data, TRAIN_HKL)


# test
images, labels, views = do('test')
data = {
        'x': images,
        'y': labels,
        'view': views,
        'classes': classes
}
hkl.dump(data, TEST_HKL)


