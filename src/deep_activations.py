
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import urllib.request
import json

#keras models
import keras
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.applications.resnet50 import ResNet50
from keras.applications.inception_v3 import InceptionV3
from keras import backend as K
K.clear_session()
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input, decode_predictions
from keras.applications.vgg19 import preprocess_input, decode_predictions
from keras.applications.resnet50 import preprocess_input, decode_predictions
from keras.applications.inception_v3 import preprocess_input, decode_predictions



def mkdir(directory):
    if not os.path.exists(directory):
        os.mkdir(directory)
        
def superpos_heatmap(img,heatmap):
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = heatmap * 0.4 + img
    result_img = image.array_to_img(superimposed_img)
    return result_img,superimposed_img
    
def get_dataset_data(dataset_name='ImageNet',log = False):
    if dataset_name=='ImageNet':
        download_url = 'https://raw.githubusercontent.com/raghakot/keras-vis/master/resources/imagenet_class_index.json'
        data_path = 'imagenet_class_index.json'
    if os.path.isfile(data_path) == False:
        urllib.request.urlretrieve(download_url,data_path)
    with open(data_path) as f:
        data = json.load(f) 
    if log == True:
        print(data)
    return data
    
def get_idx_fromlabel(label,imagenet_data):
    for key in imagenet_data:
        #print(imagenet_data[key][1])
        if imagenet_data[key][1]==label:
            idx=int(key)
    return idx
    
def get_dataset_labels(imagenet_data):
    labels=[]
    keys=[]
    synsets=[]
    for key in imagenet_data:
        #print(imagenet_data[key][1])
        labels.append(imagenet_data[key][1])
        keys.append(key)
        synsets.append(imagenet_data[key][0])
    #print(labels)
    #print(keys)
    #print(synsets)
    return labels,keys,synsets

#https://keras.io/applications/
def keras_load_model(model_name = 'VGG16', dataset_name = 'ImageNet'):
    ## weight paths
    #https://github.com/fchollet/deep-learning-models/releases
    # select weights
    if dataset_name == 'ImageNet':
        if model_name == 'VGG16':
            download_url = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels.h5'
            weights_path = 'data/vgg16_weights_tf_dim_ordering_tf_kernels.h5'
        elif model_name == 'VGG19':
            download_url = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg19_weights_tf_dim_ordering_tf_kernels.h5'
            weights_path = 'data/vgg19_weights_tf_dim_ordering_tf_kernels.h5'
        elif model_name == 'ResNet50':
            download_url = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels.h5'
            weights_path = 'data/resnet50_weights_tf_dim_ordering_tf_kernels.h5'
        elif model_name == 'InceptionV3':
            download_url = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.2/inception_v3_weights_tf_dim_ordering_tf_kernels.h5'
            weights_path = 'data/inception_v3_weights_tf_dim_ordering_tf_kernels.h5'
    #check weights file or download
    if os.path.isfile(weights_path) == False:
        urllib.request.urlretrieve(download_url,weights_path)
    #load model
    if model_name == 'VGG16':
        model = VGG16()
        #model = VGG16(weights='imagenet')
    elif model_name == 'VGG19':
        model = VGG19()
        #model = VGG19(weights='imagenet')
    elif model_name == 'ResNet50':
        model = ResNet50()
        #model = ResNet50(weights='imagenet')
    elif model_name == 'InceptionV3':
        model = InceptionV3()
        #model = InceptionV3(weights='imagenet')
    #load weights
    model.load_weights(weights_path)
    
    return model


def keras_preprocess(img,preprocess_resolution=[224, 224]):
    #keras model img preprocessing
    img = cv2.resize(img, (preprocess_resolution[0],preprocess_resolution[1]), interpolation=cv2.INTER_CUBIC)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x

def keras_classification(y):
    ## classification result (top 5), 
    #classes = y.argmax(axis=-1)
    winner_class_idx = np.argmax(y[0])
    top5_string=decode_predictions(y,top=5)[0]
    return winner_class_idx, top5_string


def keras_class(img,keras_model):
    ## classification result (top 5), 
    #classes = y.argmax(axis=-1)
    x=keras_preprocess(img,[224,224])
    y=keras_model.predict(x)
    winner_class_idx = np.argmax(y[0])
    top5_string=decode_predictions(y,top=5)[0]
    return winner_class_idx, top5_string

    
def keras_activation(x,class_idx,keras_model,layer):
    #attention, inspired from tutorial: https://www.youtube.com/watch?v=vVFxCoMZesw&feature=youtu.be
    #layers list https://github.com/fchollet/deep-learning-models
    vector = keras_model.output[:, class_idx]
    target_conv_layer = keras_model.get_layer(layer)
    grads = K.gradients(vector,target_conv_layer.output)[0]
    pooled_grads = K.mean(grads, axis=(0,1,2))
    iterate = K.function([keras_model.input],[pooled_grads,target_conv_layer.output[0]])
    pooled_grads_value, conv_layer_output_value = iterate([x])

    for i in range(512):
        conv_layer_output_value[:,:,i] *= pooled_grads_value[i]
        
    heatmap = np.mean(conv_layer_output_value, axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)
    #plt.matshow(heatmap)
    #plt.show()
    return heatmap
    
def keras_compile(keras_model):
    keras_model.compile(loss="binary_crossentropy",optimizer="adam",metrics=['accuracy'])
    return keras_model

def keras_train(keras_model,X,Y,batch_size=32, epochs=10,validation_split=0.1):
    keras_model.fit(X,Y,batch_size,epochs,validation_split) #callbacks=[tensorboard]
    return keras_model

def show_activations(img,keras_model,model_name='ResNet50',dataset_name='ImageNet',layer='activation_49'):
    x=keras_preprocess(img,[224,224])
    y=keras_model.predict(x)
    imagenet_data = get_dataset_data(dataset_name)
    for idx in range(0,len(imagenet_data)):
        label = imagenet_data[str(idx)][1]
        synset = imagenet_data[str(idx)][0]
        
        folder_results='output'
        folder_results_complete=folder_results+'/'+dataset_name+'/'+model_name+'/'+layer
        output_superpos_path = folder_results_complete +'/'+ label + '.jpg'
        output_heatmap_path = folder_results_complete +'/heatmap'+'/'+ label + '.jpg'
        if not os.path.exists(output_heatmap_path):
            heatmap= keras_activation(x,idx,keras_model,layer)
            heatmap = cv2.resize(heatmap, (img.shape[1],img.shape[0]))
            heatmap = np.uint8(255 * heatmap)
            heatmap_toplot,heatmap_tosave=superpos_heatmap(img,heatmap)
            
            #plot
            #plt.imshow(heatmap_toplot)
            #plt.show()
            mkdir(folder_results)
            mkdir(folder_results+'/'+dataset_name)
            mkdir(folder_results+'/'+dataset_name+'/'+model_name)
            mkdir(folder_results+'/'+dataset_name+'/'+model_name+'/'+layer)
            mkdir(folder_results+'/'+dataset_name+'/'+model_name+'/'+layer+'/heatmap')
            
            cv2.imwrite(output_superpos_path, heatmap_tosave)
            cv2.imwrite(output_heatmap_path, heatmap)

def keras_get_layer_properties(keras_model=keras_load_model(model_name = 'VGG16', dataset_name = 'ImageNet')):
    layer_names=[]
    layer_shapes=[]
    for layer in keras_model.layers:
        layer_names.append(layer.name)
        layer_shapes.append(layer.output_shape)
    #print(layer_names)
    return layer_names,layer_shapes

            
def show_activations_dataset(img,model_name='ResNet50',dataset_name='ImageNet',layer='activation_49'):
    keras_model = keras_load_model(model_name,dataset_name)
    show_activations(img,keras_model,model_name,dataset_name,layer)

def keras_singleclass_heatmap(img,label_name,model_name='ResNet50',dataset_name='ImageNet',layer='activation_49'):
    dataset_data = get_dataset_data(dataset_name)
    idx=get_idx_fromlabel(label_name,dataset_data) 
    keras_model = keras_load_model(model_name,dataset_name)
    x=keras_preprocess(img,[224,224])
    y=keras_model.predict(x)
    heatmap=keras_activation(x,idx,keras_model,layer)
    heatmap = cv2.resize(heatmap, (img.shape[1],img.shape[0]))
    return heatmap
    
def keras_singleclass_heatmap2(img,keras_model,label_name,dataset_name='ImageNet',layer='activation_49'):
    dataset_data = get_dataset_data(dataset_name)
    idx=get_idx_fromlabel(label_name,dataset_data) 
    x=keras_preprocess(img,[224,224])
    y=keras_model.predict(x)
    heatmap=keras_activation(x,idx,keras_model,layer)
    heatmap = cv2.resize(heatmap, (img.shape[1],img.shape[0]))
    return heatmap

def keras_singleclass_heatmap_alllayers(img,keras_model,label_name,dataset_name='ImageNet'):
    dataset_data = get_dataset_data(dataset_name)
    idx=get_idx_fromlabel(label_name,dataset_data) 
    x=keras_preprocess(img,[224,224])
    y=keras_model.predict(x)
    heatmaps=[]
    for layer in keras_model.layers:
        #print(layer.name)
        try:
            heatmap=keras_activation(x,idx,keras_model,layer.name)
        except:
            heatmap=np.zeros((img.shape[1],img.shape[0]))
        heatmap=cv2.resize(heatmap, (img.shape[1],img.shape[0]))
        heatmaps.append(heatmap)
    return heatmaps

def keras_bestclass_heatmap_alllayers(img,keras_model):
    x=keras_preprocess(img,[224,224])
    y=keras_model.predict(x)
    winner_idx,top5_string=keras_classification(y)
    heatmaps=[]
    for layer in keras_model.layers:
        print(layer.name)
        try:
            heatmap=keras_activation(x,winner_idx,keras_model,layer.name)
        except:
            heatmap=np.zeros((img.shape[1],img.shape[0]))
        heatmap=cv2.resize(heatmap, (img.shape[1],img.shape[0]))
        heatmaps.append(heatmap)
    return heatmaps,top5_string[0][1]
    
def keras_bestclass_heatmap(img,keras_model,layer='activation_49'):
    x=keras_preprocess(img,[224,224])
    y=keras_model.predict(x)
    winner_idx,top5_string=keras_class(img,keras_model)
    heatmap=keras_activation(x,winner_idx,keras_model,layer)
    heatmap = cv2.resize(heatmap, (img.shape[1],img.shape[0]))
    return heatmap
    
def test_deep_activations(image_path='images/Yarbus_scaled.jpg',label_name='abaya'):
    #read img
    img=cv2.imread(image_path)
    img=img[:,:,::-1]

    #single class heatmap prediction
    heatmap=keras_singleclass_heatmap(img,label_name,'ResNet50','ImageNet','activation_49')
    heatmap = np.uint8(255 * heatmap)
    heatmap_toplot,heatmap_tosave=superpos_heatmap(img,heatmap)
    plt.imshow(heatmap_toplot)
    plt.show()


def test_keras_class(image_path='images/Yarbus_scaled.jpg',keras_model=keras_load_model(model_name = 'VGG16', dataset_name = 'ImageNet')):
    img=cv2.imread(image_path)
    img=img[:,:,::-1]
    winner_class_idx, top5_string=keras_class(img,keras_model)
    print(winner_class_idx)
    print(top5_string)
    
    
'''
##test yarbus image
#test_deep_activations('images/Yarbus_scaled.jpg','abaya')

#test_keras_class('images_old/Yarbus_scaled.jpg',keras_load_model(model_name = 'VGG16', dataset_name = 'ImageNet'))
image_path='images_old/Yarbus_scaled.jpg'
img=cv2.imread(image_path)
img=img[:,:,::-1]
keras_model=keras_load_model(model_name = 'VGG16', dataset_name = 'ImageNet')
#heatmap=keras_bestclass_heatmap(img,keras_model,'block5_conv3')
#heatmaps=keras_singleclass_heatmap_alllayers(img,keras_model,'abaya',dataset_name='ImageNet')
heatmaps,toplabel=keras_bestclass_heatmap_alllayers(img,keras_model)

for heatmap in heatmaps:
    plt.imshow(heatmap)
    plt.show()
lnames,resols=keras_get_layer_properties(keras_model)
#print(lnames)
#print(resols)

'''


'''
#plot activations of img for all labels
show_activations_dataset(img,'ResNet50','ImageNet','activation_49')
show_activations_dataset(img,'VGG16','ImageNet','block5_conv3')

'''

