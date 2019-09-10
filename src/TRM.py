
import numpy as np
import cv2
import scipy.signal as sig

def image2featuremaps(basis,img):
    aimTemp = np.zeros((img.shape[0]-basis.shape[1]+1, img.shape[1]-basis.shape[1]+1, basis.shape[0]), dtype=np.float32)
    imgCopy = img.copy()
    #AIM kernels are ordered for RGB channels
    #so convert the image from BGR to RGB
    imgCopy = imgCopy[...,::-1]
    for f in range(basis.shape[0]):
            aimTemp[:, :, f] = sig.fftconvolve(imgCopy[:, :, 0], basis[f, :, :, 0], mode='valid')

            for c in range(1, img.shape[2]):
                temp = sig.fftconvolve(imgCopy[:, :, c], basis[f, :, :, c], mode='valid')
                aimTemp[:, :, f] += temp
    maxAIM = np.amax(aimTemp)
    minAIM = np.amin(aimTemp)

    #print('[AIM] minAIM=' + str(minAIM), 'maxAIM=' + str(maxAIM))

    #rescale image using global max and min
    for f in range(basis.shape[0]):
        aimTemp[:, :, f] -= minAIM
        aimTemp[:, :, f] /= (maxAIM - minAIM)
    
    return aimTemp

def featuremaps2smap(aimTemp):
    #border = round(basis.shape[1]/2)
    sm = np.zeros((aimTemp.shape[0], aimTemp.shape[1]), dtype=np.float32)
    numBins = 256
    div = 1/(aimTemp.shape[0]*aimTemp.shape[1])
    for f in range(aimTemp.shape[2]):
        idx = aimTemp[:, :, f]*(numBins-1)
        idx = idx.astype(int)
        #print(hists.shape)
        #hist=hists[:,f]
        hist, bin_edges = np.histogram(aimTemp[:, :, f], bins=numBins, range=[0,1], density=False)
        #plt.hist(hist,10)
        #plt.show()
        sm -= np.log(hist[idx]*div+0.000001)
    cv2.normalize(sm, sm, 0, 1, cv2.NORM_MINMAX)
    sm = cv2.GaussianBlur(sm,(31, 31), 8, cv2.BORDER_CONSTANT)
    #sm = cv2.copyMakeBorder(sm, border, border, border, border, cv2.BORDER_CONSTANT, 0)
    return sm
    
def featuremaps2rmap_old(aimTemp,hists):
    #border = round(basis.shape[1]/2)
    sm = np.zeros((aimTemp.shape[0], aimTemp.shape[1]), dtype=np.float32)
    numBins = 256
    div = 1/(aimTemp.shape[0]*aimTemp.shape[1])
    for f in range(aimTemp.shape[2]):
        idx = aimTemp[:, :, f]*(numBins-1)
        idx = idx.astype(int)
        #print(hists.shape)
        hist=hists[:,f]
        #hist, bin_edges = np.histogram(aimTemp[:, :, f], bins=numBins, range=[0,1], density=False)
        #plt.hist(hist,10)
        #plt.show()
        sm -= np.log(hist[idx]*div+0.000001)
    cv2.normalize(sm, sm, 0, 1, cv2.NORM_MINMAX)
    sm = cv2.GaussianBlur(sm,(31, 31), 8, cv2.BORDER_CONSTANT)
    #sm = cv2.copyMakeBorder(sm, border, border, border, border, cv2.BORDER_CONSTANT, 0)
    return sm

def featuremaps2rmap(aimTemp,weights):
    #border = round(basis.shape[1]/2)
    rm = np.zeros((aimTemp.shape[0], aimTemp.shape[1]), dtype=np.float32)
    numBins = 256
    div = 1/(aimTemp.shape[0]*aimTemp.shape[1])
    for f in range(aimTemp.shape[2]):
        idx = aimTemp[:, :, f]*(numBins-1)*weights[f]
        idx = idx.astype(int)
        hist, bin_edges = np.histogram(aimTemp[:, :, f], bins=numBins, range=[0,1], density=False)
        #hist=hists[:,f]
        #plt.hist(hist,10)
        #plt.show()
        rm -= np.log(hist[idx]*div+0.000001)
    cv2.normalize(rm, rm, 0, 1, cv2.NORM_MINMAX)
    rm = cv2.GaussianBlur(rm,(31, 31), 8, cv2.BORDER_CONSTANT)
    #sm = cv2.copyMakeBorder(sm, border, border, border, border, cv2.BORDER_CONSTANT, 0)
    return rm



class TRM:
    def __init__(self, h, w, settings=""):
        self.settings = settings        
        self.height = h
        self.width = w
        self.TaskRelevanceMap = None
        self.invTaskRelevanceMap = None
        
    def reset(self, h, w):
        self.TaskRelevanceMap = np.zeros((h, w), dtype=np.float32)
        self.height = h
        self.width = w
        
    def ComputeInvTRM(self):
        self.invTaskRelevanceMap=1.0-self.TaskRelevanceMap
    def setTRM(self,relevance):
        self.TaskRelevanceMap=np.resize(relevance,(self.height,self.width))
        self.ComputeInvTRM()
        
        
        
