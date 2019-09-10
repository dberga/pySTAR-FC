
import os
import cv2

def create_training_data(DATADIR,CATEGORIES):
    training_data=[]
    for category in CATEGORIES:
        path=os.path.join(DATADIR,category)
        class_num=CATEGORIES.index(category)
        for img in os.listdir(path):
            try:
                img_array=cv2.imread(os.path.join(path,img))
                #new_array=cv2.resize(img_array,(IMG_SIZE,IMG_SIZE))
                training_data.append([img_array,class_num])
            except Exception as e:
                pass
    return training_data
    
def dataset2training(DATADIR):
    CATEGORIES=[]
    for r, d, f in os.walk(DATADIR):
        for folder in d:
            CATEGORIES.append(folder)
            #CATEGORIES.append(os.path.join(r, folder))
    #for f in CATEGORIES:
    #    print(f)
    training_data=create_training_data(DATADIR,CATEGORIES)
    for features, label in training_data
        X.append(features)
        Y.append(label)
    #X=np.array(X).reshape(-1,IMG_SIZE,IMG_SIZE,1)
    return X,Y


