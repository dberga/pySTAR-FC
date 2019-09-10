#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from sklearn.decomposition import MiniBatchDictionaryLearning
from sklearn.feature_extraction.image import extract_patches_2d
from sklearn.feature_extraction.image import reconstruct_from_patches_2d
import scipy.io as sio
import matplotlib.pyplot as plt
import math
import cv2

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn import mixture as mix

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x))

def softmax_axis(x, axis=0):
    return np.apply_along_axis(softmax,0,x)
    


def printbasis(basis):
    columns=5
    rows=math.ceil(basis.shape[0]/columns)
    fig=plt.figure(figsize=(rows,columns))

    for f in range(1,basis.shape[0]+1):
            rgbbasis=basis[f-1, :, :, :]
            rgbbasis=(rgbbasis-rgbbasis.min())/(rgbbasis.max()-rgbbasis.min())
            fig.add_subplot(rows, columns, f) #old: f+1
            plt.imshow(rgbbasis)
    plt.show()
    
def plot_components(V,patch_size):
    plt.figure(figsize=(4.2, 4))
    for i, comp in enumerate(V):
        #print(comp.reshape(patch_size).shape)
        patch=comp.reshape(patch_size)
        patch=normalizep(patch)
        if len(patch_size)>2:
            plt.subplot(10,np.ceil(V.shape[0]*0.1), i + 1)
        else:
            plt.subplot(patch_size[0],patch_size[1], i + 1)
        plt.imshow(patch)
        plt.xticks(())
        plt.yticks(())
        #plt.suptitle('Dictionary',fontsize=16)
        #plt.subplots_adjust(0.08, 0.02, 0.92, 0.85, 0.08, 0.23)
    plt.show()
    
def components2basis(V,patch_size):
    basis=np.zeros((V.shape[0],patch_size))
    for i, comp in enumerate(V):
        patch=comp.reshape(patch_size)
        basis[i,:,:,:]=patch
    return basis

def loadBasis( basisMatPath):
    B = sio.loadmat(basisMatPath)['B'].astype(np.float32)
    B = np.asfortranarray(B)
    kernel_size = int(np.sqrt(B.shape[1]/3))
    #print(B.shape, kernel_size)
    basis = np.reshape(B, (B.shape[0], kernel_size, kernel_size, 3), order='F')
    
    #AIM requires correlation operation, but since scipy only has convolution available
    #we need to flip the kernels vertically and horizontally
    for i in range(basis.shape[0]):
        for j in range(basis.shape[3]):
            basis[i, :, :, j] = np.fliplr(np.flipud(basis[i, :, :, j]))
    return basis

def writeBasis(basis,basisMatPath):
    for i in range(basis.shape[0]):
        for j in range(basis.shape[3]):
            basis[i, :, :, j] = np.fliplr(np.flipud(basis[i, :, :, j]))
    kernel_size=basis.shape[1]
    Bshape=np.power(kernel_size,2)*3
    B=np.reshape(basis,(basis.shape[0],Bshape),order='F')
    sio.savemat(basisMatPath, {'B':B})

def basis2components(basis,patch_size):
    n_components=basis.shape[0]*basis.shape[3]
    vector_size=patch_size[0]*patch_size[1]
    components=np.zeros((n_components,vector_size))
    for f in range(basis.shape[0]):
        for c in range(basis.shape[3]):
            patch=basis[f,:,:,c]
            patch2=np.resize(patch,patch_size)
            component=np.reshape(patch2,(vector_size))
            components[f,:]=component
    return components

def basis2components3d(basis,patch_size):
    n_components=basis.shape[0]
    vector_size=patch_size[0]*patch_size[1]*patch_size[2]
    components=np.zeros((n_components,vector_size))
    for f in range(basis.shape[0]):
        patch=basis[f,:,:]
        patch2=np.resize(patch,patch_size)
        component=np.reshape(patch2,(vector_size))
        components[f,:]=component
    return components

def loadDico(components_,transform_algorithm='lars',kwargs={'transform_n_nonzero_coefs': 5}):
    n_components=components_.shape[0]
    dico = MiniBatchDictionaryLearning(n_components=n_components, alpha=1, n_iter=500)
    dico.set_params(transform_algorithm=transform_algorithm, **kwargs)
    dico.components_=components_
    dico.n_components_=n_components
    return dico

def normalizep(x, vmin=0, vmax=1, positive=False): #rescaling
    #normalize between 0 and 1
    xmax=x.max()
    xmin=x.min()
    x=np.divide((x-xmin),(xmax-xmin)+np.finfo(float).eps)
    #normalize between vmin and vmax
    x=vmin+((vmax-vmin)*x)
    #erase negatives
    if positive==True:
        x[x<0]=0
    return x
def normalizem(x, positive=False): #mean normalization
    xmax=x.max()
    xmin=x.min()
    xmean=np.mean(x)
    x=np.divide((x-xmean),(xmax-xmin)+np.finfo(float).eps)
    if positive==True:
        x[x<0]=0
    return x
def normalizest(x, positive=False): #standarization
    xmu=np.nanmean(x)
    xstd=np.nanstd(x)
    x=np.divide((x-xmu),xstd+np.finfo(float).eps)
    #erase negatives
    if positive==True:
        x[x<0]=0
    return x

def normalizeu(x): #unit norm rescaling
    return np.divide(x,np.linalg.norm(x)+np.finfo(float).eps)

def normalizedm(x): #rescaling
    return np.divide(x,np.mean(x)+np.finfo(float).eps)

def normalizesum(x): #energy normalization
    return np.divide(x,np.sum(x)+np.finfo(float).eps)


def vector_normalization(x, method='meanrescaling'):
    if x.shape[0] > 1:
        for idx in range(x.shape[1]):
            if method=='meanrescaling':
                x[:,idx]=normalizem(x[:,idx])
            elif method=='rescaling':
                x[:,idx]=normalizep(x[:,idx])
            elif method=='standarization':
                x[:,idx]=normalizest(x[:,idx])
            elif method=='unitrescaling':
                x[:,idx]=normalizeu(x[:,idx])
            elif method=='meannormalization':
                x[:,idx]=normalizedm(x[:,idx])
    return x

def img2data(img,patch_size,targetsize=256):
    img=img.astype(float)
    height, width=img.shape
    r = targetsize / img.shape[1]
    dim = (targetsize, int(img.shape[0] * r))
    img2 = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    data = extract_patches_2d(img2[:, :], patch_size)
    data = data.reshape(data.shape[0], -1)
    intercept = np.mean(data, axis=0)
    data -= intercept
    return data,intercept

def img2data3d(img,patch_size,targetsize=256):
    img=img.astype(float)
    height, width, channels =img.shape
    r = targetsize / img.shape[1]
    dim = (targetsize, int(img.shape[0] * r))
    if dim[1] < patch_size[1]:
        diff=patch_size[1]-dim[1]
        dim = (targetsize+diff,patch_size[1])
    img2 = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    data = extract_patches_2d(img2[:, :, :], (patch_size[0],patch_size[1]))
    data = data.reshape(data.shape[0], -1)
    intercept = np.mean(data, axis=0)
    data -= intercept
    return data,intercept

def SIndex(smap,mask):
    smap=normalizesum(smap)
    mask=normalizesum(mask)
    
    target=smap*mask
    npixWin=np.sum(mask)
    Ls=np.sum(target)
    Lb = np.sum( smap)
    Lb = (Lb- Ls)/(smap.size-npixWin)
    Ls=Ls/npixWin
    Sw=(Ls-Lb)/(Lb+np.finfo(float).eps) #weber law
    return Sw

def centdist(smap,mask):
    centroid_smap = np.argmax(smap)
    centroid_mask = np.argmax(mask)
    
    dist = np.linalg.norm(centroid_smap-centroid_mask)
    return dist
    
def plot_codes_hist(basis,hist):
	fig = plt.figure()
	axplot = fig.add_axes([0.10,0.25,0.90,0.70])
	axplot.bar(range(len(hist)),hist)
	numicons = basis.shape[0]
	#print(basis.shape)
	prop=(0.97-0.14)/numicons
	for k in range(numicons):
		axicon = fig.add_axes([0.14+prop*k,0.10,0.04,0.04])
		patch=normalizep(np.resize(basis[k,:,:,:],[21,21,3]))
		axicon.imshow(patch,interpolation='nearest')
		axicon.set_xticks([])
		axicon.set_yticks([])
	fig.show()
	plt.show()
    
def poolsamples(samplesvec,axis=1,method='avg'):
    if samplesvec.ndim>1:
        if method == 'avg':
            return  np.mean(samplesvec,axis)
        elif method == 'max':
            return np.max(samplesvec,axis)
        elif method == 'softmax':
            return softmax_axis(samplesvec,axis)
    else:
        return samplesvec
    

def maximizedist_regularize(vectormat,axis=0,method='PCA'):
    if method=='PCA':
        pca = PCA(n_components=vectormat.shape[axis])  
        vectorrec=np.squeeze(pca.fit_transform(vectormat.T)).T
    elif method=='kmeans':
        vectorrec = KMeans(n_clusters=vectormat.shape[axis]).fit_predict(vectormat.T)
    elif method=='gmm':
        vectorrec = mix.GaussianMixture(n_components=vectormat.shape[axis]).fit(vectormat.T).predict(vectormat.T)
    elif method=='avg':
        vectormean=np.nanmean(vectormat,axis)+np.finfo(float).eps
        vectorrec = np.divide(vectormat,vectormean)
    elif method=='rescaling':
        vectorrec=normalizep(vectorrec)
    return vectorrec

class Coder:
    def __init__(self):
        self.transform_algorithm='lars'
        self.kwargs={'transform_n_nonzero_coefs': 5}
        self.pool='max'
        self.normalize=0 #this will be done in LTM
        self.patch_size=(7,7,3)
        self.dico=None
        self.tmpData=None
        self.tmpIntercept=None
        
    def setDico(self,basisMatPath):
       basis= loadBasis( basisMatPath)
       components_=basis2components3d(basis,self.patch_size)
       self.dico=loadDico(components_,self.transform_algorithm,self.kwargs)
       
    def encode(self,img,mask=None):
        scale=256 #rescale image data to 256
        self.tmpData,self.tmpIntercept=img2data3d(img,self.patch_size,scale)
        code = self.dico.transform(self.tmpData)
        if mask is not None:        
            #correct codes according to mask patches
            self.tmpDataMask,self.tmpInterceptMask=img2data3d(mask,self.patch_size,scale)
            maskpatchval=normalizep(np.mean(self.tmpDataMask,1))
            code2=(code.T*maskpatchval).T
            code=code2
        #type of pooling
        if self.pool == 'avg':
            vector=np.mean(code, axis=0)
        elif self.pool == 'max':
            vector=np.max(code, axis=0)
        elif self.pool == 'softmax':
            vector=softmax_axis(code, axis=0)
        #normalize?    
        if self.normalize == 1:
            vector=normalizep(vector,0,1,False)
        elif self.normalize == 2:
            vector=normalizep(vector,0,1,True)
        return vector,code
    def encode_multiscale(self,img,mask=None):
        scales_list=[32,64,128,256]
        vector_list=np.zeros((self.dico.n_components,len(scales_list)))
        code_list=[]
        for s,scale in enumerate(scales_list):
            self.tmpData,self.tmpIntercept=img2data3d(img,self.patch_size,scales_list[s])
            code = self.dico.transform(self.tmpData)
            if mask is not None:
                ##correct codes according to mask patches
                self.tmpDataMask,self.tmpInterceptMask=img2data3d(mask,self.patch_size,scales_list[s])
                maskpatchval=normalizep(np.mean(self.tmpDataMask,1))
                code2=(code.T*maskpatchval).T
                code=code2
            if self.pool == 'avg':
                vector=np.nanmean(code, axis=0)
            elif self.pool == 'max':
                vector=np.max(code, axis=0)
            elif self.pool == 'softmax':
                vector=softmax_axis(code, axis=0)
            if self.normalize == 1:
                vector=normalizep(vector,0,1,False)
            elif self.normalize == 2:
                vector=normalizep(vector,0,1,True)
            vector_list[:,s]=vector
            code_list.append(code)
        vector_out=np.mean(vector_list,axis=1)
        return vector_out,code_list
        
    def decode(self,img,code):
        height, width, channels=img.shape
        patches = np.dot(code, self.dico.components_)
        patches += self.tmpIntercept
        patches = patches.reshape(len(self.tmpData), *self.patch_size)
        if self.dico.transform_algorithm == 'threshold':
            patches -= patches.min()
            patches /= patches.max()
        reconstruction = reconstruct_from_patches_2d(patches, (height, width,channels))
        return reconstruction
    
'''
sklearn.decomposition.sparse_encode(data,dictionary)
'''


    
    
