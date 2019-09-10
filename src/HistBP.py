
import numpy as np
import cv2
import matplotlib.pyplot as plt

####VISUAL LEARNING REPRESENTATIONS (ALTERNATIVE 0. BACKPROJECTION)

def map2hsv(map): 
        map=map[:,:,::-1]
        hsv = cv2.cvtColor(map,cv2.COLOR_BGR2HSV)
        return hsv
    
def map2hist(map): #learning cue
        map=map[:,:,::-1]
        hsv = cv2.cvtColor(map,cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv],[0, 1], None, [180, 256], [0, 180, 0, 256] )
        return hist

def hbProp (lmap,hist2): #backproject
    try:
        dst = cv2.calcBackProject([lmap],[0,1],hist2,[0,180,0,256],1)
        disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
        cv2.filter2D(dst,-1,disc,dst)
        #ret,dst = cv2.threshold(dst,50,255,cv2.THRESH_BINARY)
        dst=np.float32(dst)
        cv2.normalize(dst, dst, 0, 1, cv2.NORM_MINMAX)
        dst = cv2.GaussianBlur(dst,(31, 31), 8, cv2.BORDER_CONSTANT)
    except:
        dst = np.zeros(lmap.shape[:2],np.uint8)
    return dst

def bbox2template(boundingbox): #graph cut background-foreground segmentation
    mask = np.zeros(boundingbox.shape[:2],np.uint8)
    bgdModel = np.zeros((1,65),np.float64)
    fgdModel = np.zeros((1,65),np.float64)
    
    #rectangle discard about 5% of image
    rect = (int(boundingbox.shape[0]*0.05),int(boundingbox.shape[1]*0.05),boundingbox.shape[0]-int(boundingbox.shape[0]*0.05),boundingbox.shape[1]-int(boundingbox.shape[1]*0.05))
    #rect = (0,0,boundingbox.shape[0],boundingbox.shape[1])
    cv2.grabCut(boundingbox,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)
    
    mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
    template = boundingbox*mask2[:,:,np.newaxis]
    
    #if result is an empty image, use bounding box
    tgray=cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    tmean=np.mean(tgray,axis=(0,1))
    if tmean < 50:
        return boundingbox
    return template

def bbox2mask(boundingbox): #graph cut background-foreground segmentation
    mask = np.zeros(boundingbox.shape[:2],np.uint8)
    bgdModel = np.zeros((1,65),np.float64)
    fgdModel = np.zeros((1,65),np.float64)
    
    #rectangle discard about 5% of image
    rect = (int(boundingbox.shape[0]*0.05),int(boundingbox.shape[1]*0.05),boundingbox.shape[0]-int(boundingbox.shape[0]*0.05),boundingbox.shape[1]-int(boundingbox.shape[1]*0.05))
    #rect = (0,0,boundingbox.shape[0],boundingbox.shape[1])
    cv2.grabCut(boundingbox,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)
    
    mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
    mask = mask2[:,:,np.newaxis]
    #if result is an empty image, use bounding box
    tgray=mask
    tmean=np.mean(tgray,axis=(0,1))
    if tmean < 50:
        return np.ones((boundingbox.shape),np.float64)
    return mask
    
'''
def img2rep(img,type="hsv"):
    if type=="hsv":
        rep=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
        channels=range(rep.shape[2]-1)
        amplitudes=[180,256]
    if type=="rgb":
        rep=img
        channels=range(rep.shape[2])
        amplitudes=[256,256,256]
    elif type=="aim":
        dictionary_path="data/dictionary/31infomax950.mat"
        basis=loadBasis(dictionary_path)
        rep=image2featuremaps(basis,img)
        rep=rep[:,:,range(0,2)]
        channels=range(rep.shape[2])
        amplitudes=np.repeat(256,len(channels))
    return rep,channels,amplitudes

def featuremaps2hists(aimTemp):
    numBins = 256
    hists=np.zeros((256,aimTemp.shape[2]))
    for f in range(aimTemp.shape[2]):
        hist, bin_edges = np.histogram(aimTemp[:, :, f], bins=numBins, range=[0,1], density=False)
        #hist = cv2.calcHist([aimTemp[:, :, f]],[0, 1], None, [180, 256], [0, 180, 0, 256] )
        hists[:,f]=hist
    return hists
'''

    
    
    
