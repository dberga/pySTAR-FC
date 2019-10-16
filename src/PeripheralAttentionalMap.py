import numpy as np
import math
import cv2
import sys

class PeripheralAttentionalMap:

    def __init__(self, h, w, settings):
        self.settings = settings
        self.height = h
        self.width = w
        self.salMap = None
        self.periphMap = None

        if 'AIM' in settings.PeriphSalAlgorithm:
            sys.path.insert(0, 'contrib/AIM')
            from AIM import AIM
            self.buSal = AIM(settings.AIMBasis)
        elif 'ICF' in settings.PeriphSalAlgorithm:
            if not os.path.exists('contrib/ICF/ICF.py'):
                os.system('cd contrib; sh download_DeepGazeII_and_ICF.sh; cd ..')
            sys.path.insert(0, 'contrib/ICF')
            from ICF import ICF
            self.buSal = ICF()

        self.initPeripheralMask()

    def initPeripheralMask(self):
        self.periphMask = np.zeros((self.height,self.width), np.uint8)
        centX = round(self.height/2)
        centY = round(self.width/2)

        self.settings.pSizePix = self.settings.pSizeDeg*self.settings.pix2deg
        print(self.settings.pSizePix)
        for i in range(self.height):
            for j in range(self.width):
                rad = math.sqrt((i-centX)*(i-centX) + (j-centY)*(j-centY))
                if (rad <= self.settings.pSizePix):
                    self.periphMask[i, j] = 0
                else:
                    self.periphMask[i, j] = 1

    def computeBUSaliency(self, view):
        self.buSal.loadImage(view)
        self.salMap = self.buSal.computeSaliency()

    def computePeriphMap(self, mask):
        blurredPeriphMap = cv2.GaussianBlur(self.salMap,(11,11),0)
        self.periphMap = self.salMap.copy()
        if mask:
            self.periphMap[self.periphMask==0] = 0
