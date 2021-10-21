#importing lib
import numpy as np
import cv2
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from skimage import feature
from imutils import build_montages
from imutils import paths
import os
import pickle

def quantify_image(image):
    features=feature.hog(image,orientations=9,
                         pixels_per_cell=(10,10),cells_per_block=(2,2),
                         transform_sqrt=True,block_norm='L1')
    return features

def load_split(path):
    iamgePaths=list(paths.list_images(path))
    data=[]
    labels=[]
    for imagePath in iamgePaths:
        label=imagePath.split(os.path.sep)[-2]
        image=cv2.imread(imagePath)
        image=cv2.cvtColor(image,cv2.COLOR_BG2GRAY)
        image=cv2.resize(image,(200,200))
        image = cv2.threshold(image, 0, 255,
			cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        
        features=quantify_image(image)
        data.append(features)
        labels.append(label)
    return (np.array(data),np.array(labels))

trainingPath=r'..\dataset\spiral\training'
testingPath=r'..\dataset\spiral\testing'

(X_train, y_train) = load_split(trainingPath)
(X_test, y_test) = load_split(testingPath)
le=LabelEncoder()
y_train=le.fit_transform(y_train)
y_test=le.fit_transform(y_test)
X_train=np.reshape(X_train,(-1,1))
y_train=np.reshape(y_train,(-1,1))
X_test=np.reshape(X_test,(-1,1))
y_test=np.reshape(y_test,(-1,1))


print("[INFO] training model")
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)
pickle.dump(model,open('parkinson.pkl','wb'))

predictions=model.predict(X_test)
cm=confusion_matrix(y_test, predictions).flatten()
print(cm)
(tn,fp,fn,tp)=cm
accuracy=(tp+tn)/float(cm.sum())
print(accuracy)
