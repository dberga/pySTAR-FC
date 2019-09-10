from os import walk
from LTM import *
from TRM import *
from HistBP import *
import cv2
import numpy as np
import matplotlib.pyplot as plt
import nltk
#from deep_activations import *

 
    
class vTE:
    def __init__(self, settings=""):
        self.settings = settings
        self.task_graph = None
        self.hists_mem = None
        self.weights_mem = None
        
        try:
            self.task_specification=settings.task_text
        except:
            self.task_specification=None
            
        self.task_type=1 #sentence
    
    def get_relevance(self,LTM,img):
        if self.task_type==0: #keyword
            return self.get_kw(LTM,img,self.task_specification)
        else: #sentence
            return self.get_snt(LTM,img,self.task_specification)
    
            
    def get_kw(self,LTM,img,keyword):
        if LTM.method == 'HistogramBackprojection':
            return self.backproject_kw(LTM,img,keyword)
        elif LTM.method == 'SparseCoding':
            return self.maxcode_kw(LTM,img,keyword)
        elif LTM.method == 'DCNN':
            return self.activations_kw(LTM,img,keyword)
        
    def get_snt(self,LTM,img,sentence):
        if LTM.method == 'HistogramBackprojection':
            return self.backproject_snt(LTM,img,sentence)
        elif LTM.method == 'SparseCoding':
            return self.maxcode_snt(LTM,img,sentence)
        elif LTM.method == 'DCNN':
            return self.activations_snt(LTM,img,keyword)
        
    def activations_kw(self,LTM,img,keyword):
        relevance=keras_singleclass_heatmap2(img,keyword,LTM.keras_model_name,LTM.keras_dataset_name,LTM.keras_activation_layer)
        return relevance
        
    def activations_snt(self,LTM,img,sentence):    
        self.hists_mem,self.weights_mem=LTM.retrieve_snt(sentence)
        aimTemp=image2featuremaps(LTM.basis,img)
        #self.task_keywords,self.task_graph=text2graph(sentence)
        rels=np.zeros((aimTemp.shape[0],aimTemp.shape[1],len(self.weights_mem)))
        for w,weight in enumerate(self.weights_mem):
            self.hists_mem[w]=self.hists_mem[w]/255
            #plot_codes_hist(LTM.basis,self.hists_mem[w])
            rels[:,:,w]=featuremaps2rmap(aimTemp,self.hists_mem[w])*weight
        relevance=np.mean(rels,axis=(2))
        relevance = cv2.resize(relevance, (img.shape[1],img.shape[0]), 0, 0, cv2.INTER_AREA)
        return relevance
    def maxcode_kw(self,LTM,img,keyword):      
        self.hists_mem,self.weights_mem=LTM.retrieve_kw(keyword)
        self.hists_mem=self.hists_mem/255
        #plot_codes_hist(LTM.basis,self.hists_mem)
        aimTemp=image2featuremaps(LTM.basis,img)
        relevance=featuremaps2rmap(aimTemp,self.hists_mem)
        relevance = cv2.resize(relevance, (img.shape[1],img.shape[0]), 0, 0, cv2.INTER_AREA)
        return relevance
    
    def maxcode_snt(self,LTM,img,sentence):    
        self.hists_mem,self.weights_mem=LTM.retrieve_snt(sentence)
        aimTemp=image2featuremaps(LTM.basis,img)
        #self.task_keywords,self.task_graph=text2graph(sentence)
        rels=np.zeros((aimTemp.shape[0],aimTemp.shape[1],len(self.weights_mem)))
        for w,weight in enumerate(self.weights_mem):
            self.hists_mem[w]=self.hists_mem[w]/255
            #plot_codes_hist(LTM.basis,self.hists_mem[w])
            rels[:,:,w]=featuremaps2rmap(aimTemp,self.hists_mem[w])*weight
        relevance=np.mean(rels,axis=(2))
        relevance = cv2.resize(relevance, (img.shape[1],img.shape[0]), 0, 0, cv2.INTER_AREA)
        return relevance
    
    def backproject_kw(self,LTM,img,keyword):      
        self.hists_mem,self.weights_mem=LTM.retrieve_kw(keyword)
        hsvt = map2hsv(img)
        relevance=hbProp(hsvt,self.hists_mem)
        return relevance
    
    
    def backproject_snt(self,LTM,img,sentence):    
        self.hists_mem,self.weights_mem=LTM.retrieve_snt(sentence)
        hsvt = map2hsv(img)
        #self.task_keywords,self.task_graph=text2graph(sentence)
        rels=np.zeros((img.shape[0],img.shape[1],len(self.weights_mem)))
        for w,weight in enumerate(self.weights_mem):
            rels[:,:,w]=hbProp(hsvt,self.hists_mem[w])*weight
        relevance=np.mean(rels,axis=(2))
        return relevance
    
    #deprecated (AIM)
    def get_saliency(self,LTM,img): 
        aimTemp=image2featuremaps(LTM.basis,img)
        saliency = featuremaps2smap(aimTemp)
        saliency = cv2.resize(saliency, (img.shape[1],img.shape[0]), 0, 0, cv2.INTER_AREA)
        
        return saliency

