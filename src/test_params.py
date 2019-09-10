
import cv2
import numpy as np
import matplotlib.pyplot as plt

#from deep_activations import *
from NLP import *

import os
import shutil
import scipy.io as sio
from scipy.misc import imsave
import csv
import sys


#test functions
def tests_kw(LTM,vTE,keywords,output_path='output',test_path='images'):
    images=os.listdir(test_path)
    print(images)
    for i,imname in enumerate(images):
        ipath="".join([test_path,"/",imname])
        opath="".join([output_path,"/",imname])
        (imname_noext,ext) = os.path.splitext(imname)
        #read image
        print(ipath)
        img=cv2.imread(ipath)
        
        img=img[:,:,::-1]
        if not os.path.exists(opath):
            #load trm from keyword
            for k,keyword in enumerate(keywords):
                trm=vTE.get_kw(LTM,img,keyword)
                opath_kw="".join([output_path,"/",imname_noext,"_",keyword,ext])
                trm_out=normalizep(trm)*255
                cv2.imwrite(opath_kw,trm_out)
        
            
def tests_snt(LTM,vTE,keywords,output_path='output',test_path='images'):
    images=os.listdir(test_path)
    for i,imname in enumerate(images):
        ipath="".join([test_path,"/",imname])
        opath="".join([output_path,"/",imname])
        (imname_noext,ext) = os.path.splitext(imname)
        #read image
        img=cv2.imread(ipath)
        img=img[:,:,::-1]
        
        if not os.path.exists(opath):
            #load trm from keyword
            for s,sentence in enumerate(keywords):
                trm=vTE.get_snt(LTM,img,sentence)
                opath_snt="".join([output_path,"/",imname_noext,"_",sentence,ext])
                trm_out=normalizep(trm)*255
                cv2.imwrite(opath_snt,trm_out)
                
def tests_original(LTM,vTE,output_path='output',test_path='images'):
    images=os.listdir(test_path)
    for i,imname in enumerate(images):
        ipath="".join([test_path,"/",imname])
        opath="".join([output_path,"/",imname])
        (imname_noext,ext) = os.path.splitext(imname)
        #read image
        img=cv2.imread(ipath)
        img=img[:,:,::-1]
        
        if not os.path.exists(opath):
            smap=vTE.get_saliency(LTM,img)
            opath_original="".join([output_path,"/",imname_noext,"_","AIM",ext])
            smap_out=normalizep(smap)*255
            cv2.imwrite(opath_original,smap_out)

def tests_sindex(output_path='output',test_path='images',masks_path='masks'):
    images=os.listdir(test_path)
    for i,imname in enumerate(images):
        ipath="".join([test_path,"/",imname])
        opath="".join([output_path,"/",imname])
        mpath="".join([masks_path,"/",imname])
        (imname_noext,ext) = os.path.splitext(imname)
        swpath="".join([output_path,"/",imname,"_SIndex.txt"])

        output_images=os.listdir(output_path)
        
        for o,oname in enumerate(output_images):
            opath2="".join([output_path,"/",oname])
            swpath2="".join([output_path,"/",oname,"_SIndex.txt"])
            if imname_noext in opath2 and ".txt" not in opath2 and os.path.exists(opath2) and os.path.exists(mpath): #opath2.find(imname_noext)  != -1: #os.path.exists(opath2) and os.path.exists(mpath):
                
                smap=cv2.imread(opath2)
                mmap=cv2.imread(mpath)
                Sw=SIndex(smap,mmap)
                print(Sw)
                file = open(swpath2,"w") 
                file.write(str(Sw)) 
                file.close()

def tests_centdist(output_path='output',test_path='images',masks_path='masks'):
    images=os.listdir(test_path)
    for i,imname in enumerate(images):
        ipath="".join([test_path,"/",imname])
        opath="".join([output_path,"/",imname])
        mpath="".join([masks_path,"/",imname])
        (imname_noext,ext) = os.path.splitext(imname)
        centpath="".join([output_path,"/",imname,"_centdist.txt"])

        output_images=os.listdir(output_path)
        
        for o,oname in enumerate(output_images):
            opath2="".join([output_path,"/",oname])
            centpath2="".join([output_path,"/",oname,"_centdist.txt"])
            print(opath2)
            print(mpath)
            if imname_noext in opath2 and ".txt" not in opath2 and os.path.exists(opath2) and os.path.exists(mpath): #opath2.find(imname_noext)  != -1: #os.path.exists(opath2) and os.path.exists(mpath):
                
                smap=cv2.imread(opath2)
                mmap=cv2.imread(mpath)
                dist=centdist(smap,mmap)
                print(dist)
                file = open(centpath2,"w") 
                file.write(str(dist)) 
                file.close()

def label2imagenet(clabel):
    imagenet_data=get_dataset_data('ImageNet')
    ilabels,ikeys,isynsets=get_dataset_labels(imagenet_data)
    weights=keywordsimilarities(clabel,ilabels)
    order=np.argsort(weights)[::-1]
    ilabel=ilabels[order[0]]
    return ilabel

def label2pascal(clabel):
    clabels=['aeroplane', 'bicycle', 'bird', 'boat', 'bottle','bus', 'car', 'cat', 'chair', 'cow', 'diningtable','dog', 'horse','motorbike', 'person', 'pottedplant','sheep', 'sofa', 'train', 'tvmonitor']
    weights=keywordsimilarities(clabel,clabels)
    order=np.argsort(weights)[::-1]
    cclabel=clabels[order[0]]
    return cclabel

def labels_pascal2imagenet(lpath='data/labels_pascal2imagenet.mat'):
    
    #test_path='datasets/pascal-s/images_parsed/'
    #clabels=os.listdir(test_path)
    clabels=['aeroplane', 'bicycle', 'bird', 'boat', 'bottle','bus', 'car', 'cat', 'chair', 'cow', 'diningtable','dog', 'horse','motorbike', 'person', 'pottedplant','sheep', 'sofa', 'train', 'tvmonitor']
    
    imagenet_data=get_dataset_data('ImageNet')
    ilabels,ikeys,isynsets=get_dataset_labels(imagenet_data)
    cilabels=[]
    if not os.path.exists(lpath):
        weights2d=keywordsetsimilarities(clabels,ilabels)
        for weights in weights2d: 
            order=np.argsort(weights)[::-1]
            bestlabel=ilabels[order[0]]
            cilabels.append(bestlabel)
        sio.savemat(lpath,{"cilabels":cilabels})
    mat_content=sio.loadmat(lpath)
    cilabels=mat_content['cilabels']
    return cilabels

print('hola1')
dirpath = os.getcwd()
output_folder="".join([dirpath,os.sep,"output"])
if not os.path.exists(output_folder):
    os.mkdir(output_folder)
print('hola2')
from LTM import LTM
from vTE import vTE
from Coder import normalizep
from Coder import normalizesum
from Coder import SIndex
from Coder import centdist
print('hola3')

#training paths

try:
    training_boxes_path=sys.argv[4] #pascal-s/boundingboxes
    training_boxes_masks_path=sys.argv[5] #pascal-s/boundingboxes_mask
    
except:
    training_boxes_path="datasets/pascal-s/boundingboxes/"#"data/training_boundingboxes/"
    training_boxes_masks_path="datasets/pascal-s/boundingboxes_mask"#"data/training_boundingboxes_masks/"    
    
#print(training_boxes_path)
#print(training_boxes_masks_path)
    
#tests
try:
    keywords_test=[sys.argv[1]] #aeroplane    
    test_path=sys.argv[2] #pascal-s/images_parsed/aeroplane
    masks_path=sys.argv[3] #pascal-s/masks
    #for k,keyword in enumerate(keywords_test):
    #    sentences_test[k]="".join(["look for ",keyword])
except:
    test_path='images_old'
    masks_path='masks'
    keywords_test=['person']#os.listdir(training_boxes_path)
    #for k,keyword in enumerate(keywords_test):
    #    sentences_test[k]="".join(["look for ",keyword])
    
#print(test_path)
#print(masks_path)
#print(keywords_test)

print('hola4')

#init LTM and vTE instances
mymem=LTM()
print('loaded mem')
taskexecutive=vTE()
print('loaded vte')
mymem.training_boxes_path=training_boxes_path
mymem.training_boxes_masks_path=training_boxes_masks_path
    
'''
#testing methods Histogram Backprojection
mymem.method='HistogramBackprojection'
mpath="".join([output_folder,"/",mymem.method])

if not os.path.exists(mpath):
    os.mkdir(mpath)
                    
if os.path.exists(mymem.histograms_path):
    shutil.rmtree(mymem.histograms_path)
    os.mkdir(mymem.histograms_path)
else:
    os.mkdir(mymem.histograms_path)
    

mymem.learn()
    
tests_kw(mymem,taskexecutive,keywords_test,mpath,test_path)
#tests_snt(mymem,taskexecutive,sentences_test,mpath,test_path)
tests_original(mymem,taskexecutive,mpath,test_path)

tests_sindex(mpath,test_path,masks_path)
tests_centdist(mpath,test_path,masks_path)
'''

#testing methods Sparse Coding
mymem.method='SparseCoding'
#transform_methods=['lars','omp','threshold','lasso_lars', 'lasso_cd']
transform_methods=['omp']
#pool_methods=['max','avg'] 
pool_methods=['max'] 
#dictionaries= os.listdir("data/dictionary")
dictionaries=['31infomax950.mat','V1_braincorp.mat','alexnet_flachot_1.mat']
#normalization_methods=['meanrescaling','rescaling','standarization','unitrescaling','meannormalization']
normalization_methods=['standarization']


mpath="".join([output_folder,"/",mymem.method])
if not os.path.exists(mpath):
    os.mkdir(mpath)
for t,transform_algorithm in enumerate(transform_methods):
    tpath="".join([mpath,"/",transform_algorithm])
    if not os.path.exists(tpath):
        os.mkdir(tpath)
    for p,pool in enumerate(pool_methods):
        ppath="".join([mpath,"/",transform_algorithm,"/",pool])
        if not os.path.exists(ppath):
            os.mkdir(ppath)
        for d,dictionary in enumerate(dictionaries):
            dpath="".join([mpath,"/",transform_algorithm,"/",pool,"/",dictionary])
            if not os.path.exists(dpath):
                os.mkdir(dpath)
            for n,normalization_method in enumerate(normalization_methods):       
                npath="".join([mpath,"/",transform_algorithm,"/",pool,"/",dictionary,"/",normalization_method])
                if not os.path.exists(npath):
                    os.mkdir(npath)
                    
                print(npath)
                
                #change parameters
                mymem.SparseCoder.pool=pool
                mymem.SparseCoder.transform_algorithm=transform_algorithm
                mymem.normalization_method=normalization_method
                mymem.dictionary_path=dpath
                
                #remove and create current learning representations folder
                #if os.path.exists(mymem.codebook_path):
                    #shutil.rmtree(mymem.codebook_path)
                    #os.mkdir(mymem.codebook_path)
                    ##print("using learned codebook folder")
                #else:
                #    os.mkdir(mymem.codebook_path)
                
                #learn represenations
                mymem.learn()
                
                #test images
                tests_kw(mymem,taskexecutive,keywords_test,npath,test_path)
                #tests_snt(mymem,taskexecutive,sentences_test,npath,test_path)
                tests_original(mymem,taskexecutive,npath,test_path)
                
                #print saliency index
                #tests_sindex(npath,test_path,masks_path)
                #tests_centdist(npath,test_path,masks_path)

'''
method='DCNN'
keras_model_names=['VGG16'] #'VGG16','VGG19','ResNet50','InceptionV3'
keras_dataset_names=['ImageNet']
#keras_activation_layers=['block5_conv3']


#for k,keyword in enumerate(keywords_test):
keyword=keywords_test[0]
ilabel=label2imagenet(keyword)
        
mpath="".join([output_folder,os.sep,method])
if not os.path.exists(mpath):
    os.mkdir(mpath)
for t,keras_model_name in enumerate(keras_model_names):
    tpath="".join([mpath,os.sep,keras_model_name])
    if not os.path.exists(tpath):
        os.mkdir(tpath)
    for p,dataset_name in enumerate(keras_dataset_names):
        ppath="".join([mpath,os.sep,keras_model_name,os.sep,dataset_name])
        if not os.path.exists(ppath):
            os.mkdir(ppath)        
        
        keras_model=keras_load_model(keras_model_name,dataset_name)
        keras_activation_layers,keras_activation_resolutions=keras_get_layer_properties(keras_model)
        
        
        images=os.listdir(test_path)
        for i,imname in enumerate(images):
            #print(imname)
            
            ipath="".join([test_path,os.sep,imname])
            (imname_noext,ext) = os.path.splitext(imname)
            #read image                
            img=cv2.imread(ipath)
            img=img[:,:,::-1]
            #load trm from keyword
            
                
            #trm=vTE.get_kw(LTM,img,keyword)
            
            #for d,layer in enumerate(keras_activation_layers):
            #    dpath="".join([mpath,"/",keras_model_name,"/",dataset_name,"/",layer])
            #    if not os.path.exists(dpath):
            #        os.mkdir(dpath)
            #    print(dpath)
            
            #opath_kw="".join([ppath,"/",imname_noext,"_",keyword,ext])
            #print(opath_kw)
            
            #dataset_data=get_dataset_data(dataset_name)
            #dataset_labels,dataset_keys,dataset_synsets= get_dataset_labels(dataset_data)                        
            #label=keyword2label(keyword,dataset_labels)
            #label='bell_pepper'     
            #print(label)
            #print(opath_kw)
            #trm=keras_singleclass_heatmap2(img,keras_model,label,dataset_name,layer)
            #trm_out=normalizep(trm)*255         
            #cv2.imwrite(opath_kw,trm_out)
            
            #show_activations(img,keras_model,keras_model_name,dataset_name,layer)
            #heatmaps,toplabel=keras_bestclass_heatmap_alllayers(img,keras_model)
            last_layer_img_path="".join([ppath,os.sep,keras_activation_layers[-1:][0],os.sep,keyword,os.sep,imname_noext,ext])
            
            if not os.path.exists(last_layer_img_path):
                heatmaps=keras_singleclass_heatmap_alllayers(img,keras_model,ilabel,dataset_name)
                
                l=0
                for layer in keras_activation_layers:
                    opath_lr_folder="".join([ppath,os.sep,layer])
                    if not os.path.exists(opath_lr_folder):
                        os.mkdir(opath_lr_folder)
                    opath_lr_kw_folder="".join([opath_lr_folder,os.sep,keyword])
                    if not os.path.exists(opath_lr_kw_folder):
                        os.mkdir(opath_lr_kw_folder)
                    opath_lr_kw="".join([opath_lr_kw_folder,os.sep,imname_noext,ext])
                        
                    print(opath_lr_kw)
                    heatmap_out=normalizep(heatmaps[l])*255
                    cv2.imwrite(opath_lr_kw,heatmap_out)#imsave
                    l=l+1
                
                #for heatmap in heatmaps:
                #    plt.imshow(heatmap)
                #    plt.show()
                
'''

            