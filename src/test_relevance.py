
import cv2
import numpy as np
import matplotlib.pyplot as plt

from LTM import LTM
from vTE import vTE
from Coder import normalizep
from Coder import normalizesum
from Coder import SIndex
from Coder import centdist
from deep_activations import *


import os
from os import walk
import shutil
import scipy.io as sio
import csv
import sys

def tests_sparsecodes(dictionary_path='data/dictionary/31infomax950.mat',images_path='datasets/pascal-s/images_parsed',bbox_path='datasets/pascal-s/boundingboxes', masks_path='datasets/pascal-s/boundingboxes_mask'):
    images=os.listdir(images_path)
    print(images)
    '''
    (imname_noext,ext) = os.path.splitext(imname)
    img=cv2.imread(ipath)
    mask=cv2.imread(ipath)

    SparseCoder=Coder()
    SparseCoder.setDico(dictionary_path)
    normalization_method='unitrescaling'
    basis=loadBasis(dictionary_path)
    
    bbox_img=cv2.imread(bbox_path)
    bbox_img=bbox_img[:,:,::-1]
                                                                
    if os.path.exists(mask_path):
        mask_img=cv2.imread(mask_path)
        vector,code=SparseCoder.encode_multiscale(bbox_img,mask_img)
    else:
        vector,code=self.SparseCoder.encode_multiscale(bbox_img)
    hist=vector
    cv2.normalize(hist,hist,0,255,cv2.NORM_MINMAX)
    hists[:,idx]=hist
    hists=poolsamples(hists,1,'avg')
    '''

images_path='datasets/pascal-s/images_parsed'
bbox_path='datasets/pascal-s/boundingboxes'
masks_path='datasets/pascal-s/boundingboxes_mask'
for (dirpath, dirnames, filenames) in walk(images_path):
    print(dirnames[0])
    for f in dirnames:
        category,ext=os.path.splitext(f)
        for (dirpath2, dirnames2, filenames2) in walk("".join([bbox_path,f])):
            hists=np.zeros((self.basis.shape[0],len(filenames2)))
            for idx,instance in enumerate(filenames2):
                bbox_path="".join([bbox_path,category,"/",instance])
                
                            
                            
                            
