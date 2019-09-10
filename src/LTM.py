

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import os
from os import walk
import cv2
from NLP import *
from HistBP import *
from Coder import *
from TRM import *
#from deep_activations import *
    
        
 #### LONG TERM MEMORY CLASS
 
class LTM:
    def __init__(self, settings=""):
        self.settings = settings       
        self.method='SparseCoding' 
        self.graphs_path="data/graphs/"
        self.dictionary_path="data/dictionary/31infomax950.mat" #31infomax950,V1_braincorp
        self.basis=loadBasis(self.dictionary_path)
        self.training_boxes_path="data/training_boundingboxes/"
        self.training_boxes_masks_path="data/training_boundingboxes_masks/"
        
        #histogram backprojection
        self.training_templates_path="data/training_templates/"
        self.histograms_path="data/histograms/"
        
        #sparse coding
        self.codebook_path="data/codebook/"
        self.SparseCoder=Coder()
        self.SparseCoder.setDico(self.dictionary_path)
        self.normalization_method='unitrescaling'
        
        #deep learning models
        #self.keras_model_name='VGG16'
        #self.keras_dataset_name='ImageNet'
        #self.keras_activation_layer='activation_49'
        #self.keras_model=keras_load_model(self.keras_model_name,self.keras_dataset_name)

    #main functions
    def learn(self):      
        if self.method == 'HistogramBackprojection':
            #self.generateTemplates()
            self.generateMasks()
            self.generateTemplatesfromMasks()
            self.learn_hb()
        elif self.method == 'SparseCoding':
            self.learn_sc()
    def retrieve_kw(self,keyword):
        if self.method == 'HistogramBackprojection':
            hists,categoryweights=self.retrieve_kw_hb(keyword)
        elif self.method == 'SparseCoding':
            hists,categoryweights=self.retrieve_kw_sc(keyword)
        elif self.method == 'DCNN':
            hists,categoryweights=self.retrieve_kw_dcnn(keyword)
        return hists,categoryweights
    def retrieve_snt(self,sentence):
        if self.method == 'HistogramBackprojection':
            mhists,categoryweights=self.retrieve_snt_hb(sentence)
        elif self.method == 'SparseCoding':
            mhists,categoryweights=self.retrieve_snt_sc(sentence)
        return mhists,categoryweights
    
    #secondary functions
    def getsymWeights(self,sentence):
         categories = os.listdir(self.training_boxes_path)
         keywords,G=text2graph(sentence)
         weights=categories2symweigths(keywords,categories)
         weights[weights<0.5]=0.0 #discard half of non-semantically similar concepts (Choi & Kim 2012)
         return weights
     
    def store_sc(self,keyword,hists): 
        if not os.path.exists(self.codebook_path):
            os.mkdir(self.codebook_path)
        kpath="".join([self.codebook_path,keyword,".mat"])
        if os.path.isfile(kpath):
            os.remove(kpath)
        sio.savemat(kpath,{"hists":hists})
        return hists
    
    def store_hb(self,keyword,hists):
        if not os.path.exists(self.histograms_path):
            os.mkdir(self.histograms_path)      
        kpath="".join([self.histograms_path,keyword,".mat"])
        if os.path.isfile(kpath):
            os.remove(kpath)
        sio.savemat(kpath,{"hists":hists})
        return hists
    
    def maximize_codes(self):
        for (dirpath, dirnames, filenames) in walk(self.codebook_path):
            n_categories=len(filenames)
            n_features=self.basis.shape[0]
            cfvector=np.zeros((n_categories,n_features))
            for idx, f in enumerate(filenames): #load files to vectormat
                category,ext=os.path.splitext(f)
                kpath="".join([self.codebook_path,category,".mat"])
                mat_content=sio.loadmat(kpath)
                hists=np.float32(mat_content['hists'])
                hists=np.squeeze(hists)
                cfvector[idx,:]=hists
            #maximize distance between feature results, here
            cfvector_rec=vector_normalization(cfvector, method=self.normalization_method)
            cfvector_rec=normalizep(cfvector_rec,0,255,False)
            for idx, f in enumerate(filenames): #save vectormat to files
                category,ext=os.path.splitext(f)
                kpath="".join([self.codebook_path,category,".mat"])
                hists=cfvector_rec[idx,:]
                self.store_sc(category,hists)
                
    def generateTemplates(self):
        for (dirpath, dirnames, filenames) in walk(self.training_boxes_path):
            for f in dirnames:
                category,ext=os.path.splitext(f)
                bpath="".join([self.training_boxes_path,category])
                tpath="".join([self.training_templates_path,category])
                if not os.path.exists(tpath):
                    print(''.join(['generating templates for ',category]))
                    os.mkdir(tpath)
                    for (dirpath2, dirnames2, filenames2) in walk("".join([self.training_boxes_path,f])):
                        for idx,instance in enumerate(filenames2):
                            bbox_img=cv2.imread("".join([self.training_boxes_path,category,"/",instance]))
                            bbox_img=bbox_img[:,:,::-1]
                            template_img=bbox2template(bbox_img)
                            cv2.imwrite("".join([self.training_templates_path,category,"/",instance]),template_img)

    def generateMasks(self):
        for (dirpath, dirnames, filenames) in walk(self.training_boxes_path):
            for f in dirnames:
                category,ext=os.path.splitext(f)
                bpath="".join([self.training_boxes_path,category])
                mpath="".join([self.training_boxes_masks_path,category])
                if not os.path.exists(mpath):
                    print(''.join(['generating masks for ',category]))
                    os.mkdir(mpath)
                    for (dirpath2, dirnames2, filenames2) in walk("".join([self.training_boxes_path,f])):
                        for idx,instance in enumerate(filenames2):
                            bbox_img=cv2.imread("".join([self.training_boxes_path,category,"/",instance]))
                            #bbox_img=bbox_img[:,:,::-1]
                            mask_img=bbox2mask(bbox_img)*255
                            print(mask_img)
                            cv2.imwrite("".join([self.training_boxes_masks_path,category,"/",instance]),mask_img)
    
    def generateTemplatesfromMasks(self):
        for (dirpath, dirnames, filenames) in walk(self.training_boxes_masks_path):
            for f in dirnames:
                category,ext=os.path.splitext(f)
                bpath="".join([self.training_boxes_path,category])
                mpath="".join([self.training_boxes_masks_path,category])
                tpath="".join([self.training_templates_path,category])
                if not os.path.exists(self.training_templates_path):
                    os.mkdir(self.training_templates_path)
                if not os.path.exists(tpath):
                    print(''.join(['generating templates for ',category]))
                    os.mkdir(tpath)
                    for (dirpath2, dirnames2, filenames2) in walk(mpath):
                        for idx,instance in enumerate(filenames2):
                            bbox_img=cv2.imread("".join([self.training_boxes_path,category,"/",instance]))
                            #bbox_img=bbox_img[:,:,::-1]
                            mask_img= cv2.cvtColor(cv2.imread("".join([self.training_boxes_masks_path,category,"/",instance])),cv2.COLOR_BGR2GRAY)/255
                            template_img=bbox_img
                            for channel in range(template_img.shape[2]):
                                template_img[:,:,channel]=np.multiply(template_img[:,:,channel],mask_img)
                            cv2.imwrite("".join([self.training_templates_path,category,"/",instance]),template_img)
                            
                        
    def learn_hb(self):      
        for (dirpath, dirnames, filenames) in walk(self.training_templates_path):
            for f in dirnames:
                category,ext=os.path.splitext(f)
                kpath="".join([self.histograms_path,category,".mat"])
                if not os.path.exists(kpath):
                    print(''.join(['learning ',category]))
                    for (dirpath2, dirnames2, filenames2) in walk("".join([self.training_templates_path,f])):
                        hists=np.zeros((180,256,len(filenames2)))
                        for idx,instance in enumerate(filenames2):
                            bbox_img=cv2.imread("".join([self.training_templates_path,category,"/",instance]))
                            bbox_img=bbox_img[:,:,::-1]
                            hist=map2hist(bbox_img)
                            cv2.normalize(hist,hist,0,255,cv2.NORM_MINMAX)
                            hists[:,:,idx]=hist
                        hists=poolsamples(hists,2,'avg')
                        self.store_hb(category,hists)
                else:
                    print(''.join(['already learned ',category]))
                    
    def learn_sc(self):      
        for (dirpath, dirnames, filenames) in walk(self.training_boxes_path):
            for f in dirnames:
                category,ext=os.path.splitext(f)
                kpath="".join([self.codebook_path,category,".mat"])
                if not os.path.exists(kpath):
                    print(''.join(['learning ',category]))
                    for (dirpath2, dirnames2, filenames2) in walk("".join([self.training_boxes_path,f])):
                        hists=np.zeros((self.basis.shape[0],len(filenames2)))
                        for idx,instance in enumerate(filenames2):
                            print(instance)
                            bbox_path="".join([self.training_boxes_path,category,"/",instance])
                            mask_path="".join([self.training_boxes_masks_path,category,"/",instance])
                            bbox_img=cv2.imread(bbox_path)
                            bbox_img=bbox_img[:,:,::-1]
                                                                                        
                            if os.path.exists(mask_path):
                                mask_img=cv2.imread(mask_path)
                                #vector,code=self.SparseCoder.encode_multiscale(bbox_img,mask_img)
                                vector,code=self.SparseCoder.encode(bbox_img,mask_img)
                            else:
                                #vector,code=self.SparseCoder.encode_multiscale(bbox_img)
                                vector,code=self.SparseCoder.encode(bbox_img)
                            hist=vector
                            cv2.normalize(hist,hist,0,255,cv2.NORM_MINMAX)
                            hists[:,idx]=hist
                        hists=poolsamples(hists,1,'avg')
                        self.store_sc(category,hists)
                else:
                    print(''.join(['already learned ',category]))
        self.maximize_codes()
    
    def retrieve_kw_hb(self,keyword):
        print(''.join(['retrieving ',keyword]))
        hists=np.array([])
        for (dirpath, dirnames, filenames) in walk(self.histograms_path):
                for f in filenames:
                    (fname,ext) = os.path.splitext(f)
                    if fname == keyword:
                        kpath="".join([self.histograms_path,keyword,".mat"])
                        mat_content=sio.loadmat(kpath)
                        hists=np.float32(mat_content['hists'])
                        hists=np.squeeze(hists)
        categoryweights=self.getsymWeights(keyword)
        return hists,categoryweights
    
    def retrieve_snt_hb(self,sentence):
        print(''.join(['retrieving ',sentence]))
        keywords,graph=text2graph(sentence)
        categories = os.listdir(self.training_boxes_path)
        mhists=[]
        hists=np.array([])
        for k,keyword in enumerate(categories):
            for (dirpath, dirnames, filenames) in walk(self.histograms_path):
                for f in filenames:
                    (fname,ext) = os.path.splitext(f)
                    if fname == keyword:
                        kpath="".join([self.histograms_path,fname,".mat"])
                        mat_content=sio.loadmat(kpath)
                        hists=np.float32(mat_content['hists'])
                        hists=np.squeeze(hists)
                        #print(hists.shape)
                        mhists.append(hists)
        categoryweights=self.getsymWeights(sentence)
        return mhists,categoryweights
    
    def retrieve_kw_sc(self,keyword):
        print(''.join(['retrieving ',keyword]))
        hists=np.array([])
        for (dirpath, dirnames, filenames) in walk(self.codebook_path):
                for f in filenames:
                    (fname,ext) = os.path.splitext(f)
                    if fname == keyword:
                        kpath="".join([self.codebook_path,keyword,".mat"])
                        mat_content=sio.loadmat(kpath)
                        hists=np.float32(mat_content['hists'])
                        hists=np.squeeze(hists)
        categoryweights=self.getsymWeights(keyword)
        return hists,categoryweights
        
    def retrieve_snt_sc(self,sentence):
        print(''.join(['retrieving ',sentence]))
        keywords,graph=text2graph(sentence)
        categories = os.listdir(self.training_boxes_path)
        mhists=[]
        hists=np.array([])
        for k,keyword in enumerate(categories):
            for (dirpath, dirnames, filenames) in walk(self.codebook_path):
                for f in filenames:
                    (fname,ext) = os.path.splitext(f)
                    if fname == keyword:
                        kpath="".join([self.codebook_path,fname,".mat"])
                        mat_content=sio.loadmat(kpath)
                        hists=np.float32(mat_content['hists'])
                        hists=np.squeeze(hists)
                        #print(hists.shape)
                        mhists.append(hists)
                        #print(hists)
        categoryweights=self.getsymWeights(sentence)
        return mhists,categoryweights
    
    
    def retrieve_kw_dcnn(self,sentence):
        print(''.join(['retrieving ',sentence]))
        keywords,graph=text2graph(sentence)
        #categories = os.listdir(self.training_boxes_path)
        #hists=np.array([])
        categoryweights=self.getsymWeights(sentence)
        return categoryweights
    
    
