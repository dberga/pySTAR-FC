import numpy as np
import time
import math
import cv2
import sys
import os
from PIL import Image

class CentralAttentionalMap:
    def __init__(self, h, w, settings):
        self.height = h
        self.width = w
        self.settings = settings
        self.centralMask = None
        self.centralMap = None
        self.cv2pil = False
        if 'DeepGazeII' in settings.CentralSalAlgorithm:
            if not os.path.exists('contrib/DeepGazeII/DeepGazeII.py'):
                os.system('cd contrib; sh download_DeepGazeII_and_ICF.sh; cd ..')
            sys.path.insert(0, 'contrib/DeepGazeII')
            from DeepGazeII import DeepGazeII
            self.buSal = DeepGazeII()
        if 'SALICON' in settings.CentralSalAlgorithm:
            if not os.path.exists('contrib/SALICON/Salicon.py'):
                os.system('cd contrib; sh download_SALICON_and_SALICONtf.sh; cd ..')
            sys.path.insert(0, 'contrib/SALICON')
            from Salicon import Salicon
            self.buSal = Salicon(prototxtpath='contrib/SALICON/salicon.prototxt', model='contrib/SALICON/salicon_osie3.caffemodel')
        if 'SALICONtf' in settings.CentralSalAlgorithm:
            if not os.path.exists('contrib/SALICONtf/src/Salicon.py'):
                os.system('cd contrib; sh download_SALICON_and_SALICONtf.sh; cd ..')
            sys.path.insert(0, 'contrib/SALICONtf/src')
            from SALICONtf import SALICONtf
            self.buSal = SALICONtf(weights='contrib/SALICONtf/models/model_lr0.01_loss_crossentropy.h5')
            self.cv2pil = True
        self.initCentralMask()

    def initCentralMask(self):
        self.centralMask = np.zeros((self.height,self.width), np.uint8)
        centX = round(self.height/2)
        centY = round(self.width/2)

        self.settings.cSizePix = self.settings.cSizeDeg*self.settings.pix2deg

        for i in range(self.height):
            for j in range(self.width):
                rad = math.sqrt((i-centX)*(i-centX) + (j-centY)*(j-centY))
                if (rad <= self.settings.cSizePix):
                    self.centralMask[i, j] = 1
                else:
                    self.centralMask[i, j] = 0


    def centralDetection(self, view):
        
        if self.cv2pil:
            view_img = Image.fromarray((view[:, :, ::-1]*255).astype(np.uint8)) #convert image from cv2 to PIL format
        else:
            view_img = view*255
        #SALICON works on images with range [0, 255]
        self.centralMap = self.buSal.compute_saliency(img=view_img)
        cv2.normalize(self.centralMap, self.centralMap, 0, 1, cv2.NORM_MINMAX)

    def maskCentralDetection(self):
        self.centralMap[self.centralMask == 0] = 0
        # cv2.imshow('image',self.centralMap)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
