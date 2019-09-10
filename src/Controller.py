from os import listdir
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.gridspec as gridspec
import time
import scipy.io as sio

from Environment import Environment
from Eye import Eye
from PeripheralAttentionalMap import PeripheralAttentionalMap
from CentralAttentionalMap import CentralAttentionalMap
from ConspicuityMap import ConspicuityMap
from PriorityMap import PriorityMap
from FixationHistoryMap import FixationHistoryMap
from TRM import TRM
from vTE import vTE
from LTM import LTM

class Controller:
    def __init__(self, settings):
        self.env = None
        self.eye = None

        self.settings = settings
        self.imageList = []

        #save results
        self.saveResults = False
        if self.settings.saveFix:
            if os.path.exists(self.settings.saveDir):
                if self.settings.overwrite:
                    self.saveResults = True
            else:
                os.makedirs(self.settings.saveDir)
                self.saveResults = True
        print(self.saveResults)
    #get input images
    def getInputImages(self):
        if self.settings.input:
            self.imageList.append(self.settings.input)
        else:
            #list all images in the directory
            self.imageList = [f for f in listdir(self.settings.batch) if any(f.endswith(ext) for ext in ['jpg', 'bmp', 'png', 'gif']) ]


    def setup(self, imgPath):
        imgName, ext = os.path.splitext(os.path.basename(imgPath))
        self.imgName = imgName
        self.env = Environment(self.settings)
        self.env.loadStaticStimulus(self.settings.batch + '/' + imgPath)
        self.eye = Eye(self.settings, self.env)
        
        self.periphMap = PeripheralAttentionalMap(self.env.height, self.env.width, self.settings)
        self.centralMap = CentralAttentionalMap(self.env.height, self.env.width, self.settings)
        self.conspMap = ConspicuityMap(self.env.height, self.env.width, self.settings)
        self.priorityMap = PriorityMap(self.env.height, self.env.width, self.settings)
        self.fixHistMap = FixationHistoryMap(self.env.height, self.env.width, self.env.hPadded, self.env.wPadded, self.settings)
        
        self.LongTermMemory = LTM(self.settings)
        self.visualTaskExecutive = vTE(self.settings)
        self.TaskRelevanceMap = TRM(self.env.height, self.env.width, self.settings)
        
        if self.settings.task_relevance == 1:
            #learn representations if not done previously
            self.LongTermMemory.learn()
            #get task relevance (initial)
            self.TaskRelevanceMap.setTRM(self.visualTaskExecutive.get_relevance(self.LongTermMemory,self.env.scene))
        
    #computes fixations for each image and each subject
    def run(self):
        self.getInputImages()
        for imgPath in self.imageList:

            for i in range(self.settings.numSubjects):
                self.setup(imgPath)
                self.computeFixations()

                if self.saveResults:
                    currentSaveDir = self.settings.saveDir
                    if not os.path.exists(currentSaveDir):
                        os.makedirs(currentSaveDir)
                    #self.fixHistMap.dumpFixationsToMat('{}/fixations_{}.mat'.format(currentSaveDir, self.imgName, i))
                    cv2.imwrite('{}/fixations_{}.png'.format(currentSaveDir, self.imgName), self.env.sceneWithFixations.astype(np.uint8))
                    
                    #cv2.imwrite('{}/conspMap_{}.png'.format(currentSaveDir, self.imgName), self.conspMap.conspMap)
                    #cv2.imwrite('{}/priorityMap_{}.png'.format(currentSaveDir, self.imgName), self.priorityMap.priorityMap)
                    #cv2.imwrite('{}/fixHistMap_{}.png'.format(currentSaveDir, self.imgName), self.fixHistMap.fixHistMap)
                    
    def computeFixations(self):

        if self.settings.visualize:
            fig = plt.figure(1, figsize=(13,7), facecolor='white')
            gs = gridspec.GridSpec(2, 3)
            plt.show(block=False)
            plt.ion()

        for i in range(self.settings.maxNumFixations):
            print('fixation {}'.format(i))
            if self.saveResults:
                currentSaveDir = self.settings.saveDir
                if not os.path.exists(currentSaveDir):
                    os.makedirs(currentSaveDir)
                    
            self.eye.viewScene()

            self.periphMap.computeBUSaliency(self.eye.viewFov)
            self.periphMap.computePeriphMap(self.settings.blendingStrategy==1)

            self.centralMap.centralDetection(self.eye.viewFov)
            self.centralMap.maskCentralDetection()

            self.conspMap.computeConspicuityMap(self.periphMap.periphMap, self.centralMap.centralMap) 
            
            self.priorityMap.computeNextFixationDirection(self.periphMap.periphMap, self.centralMap.centralMap, self.fixHistMap.getFixationHistoryMap())
            
            prevGazeCoords = self.eye.gazeCoords
            self.eye.setGazeCoords(self.priorityMap.nextFixationDirection)

            self.env.drawFixation(self.eye.gazeCoords.astype(np.int32))

            self.fixHistMap.decayFixations()
            self.fixHistMap.saveFixationCoords(prevGazeCoords)
            
            
            if self.settings.visualize:
                self.add_subplot(cv2.cvtColor(self.eye.viewFov, cv2.COLOR_BGR2RGB), 'Foveated View', gs[0,0])
                self.add_subplot(self.periphMap.periphMap, 'Peripheral Map', gs[0,1])
                self.add_subplot(self.centralMap.centralMap, 'Central Map', gs[1,0])
                self.add_subplot(self.priorityMap.priorityMap, 'Priority Map', gs[1,1])
                self.add_subplot(cv2.cvtColor(self.env.sceneWithFixations.astype(np.uint8), cv2.COLOR_BGR2RGB), 'Image: {} \n Fixation #{}/{}'.format(self.imgName, i, self.settings.maxNumFixations), gs[:,-1])
                if i == 0:
                    gs.tight_layout(fig)
                fig.canvas.draw()
            if self.saveResults:   
                self.fixHistMap.dumpFixationsToMat('{}/{}.mat'.format(currentSaveDir, self.imgName, i))
            
    def add_subplot(self, img, title, plot_idx):
        ax = plt.subplot(plot_idx)
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])
        ax.set_title(title, fontsize=10)
        ax.set_xlabel('[{:10.3f}, {:10.3f}]'.format(np.min(img), np.max(img)))
        plt.imshow(img)
