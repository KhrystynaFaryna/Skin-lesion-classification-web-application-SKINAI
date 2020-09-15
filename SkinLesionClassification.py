#FINAL
# Import the libraries
import numpy as np
import keras
from keras import backend as K
from keras.layers.core import Dense, Dropout
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pyplot as plt
from operator import itemgetter
import collections 
import itertools
# Skin Cancer Dataset Preprocessing

# Import the libraries
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator
import os
from sklearn.model_selection import train_test_split
import shutil

# Create a new directory for the images
base_dir = 'base_dir'
#os.mkdir(base_dir)

# Training file directory
train_dir = os.path.join(base_dir, 'train_dir')
#os.mkdir(train_dir)

# Validation file directory
val_dir = os.path.join(base_dir, 'val_dir')
#os.mkdir(val_dir)


# Create a MobileNet model
mobile = keras.applications.mobilenet.MobileNet()

# See a summary of the layers in the model
#mobile.summary()

# Modify the model
# Exclude the last 5 layers of the model
x = mobile.layers[-6].output
# Add a dropout and dense layer for predictions
x = Dropout(0.25)(x)
predictions = Dense(7, activation='softmax')(x)

# Create a new model with the new outputs
model = Model(inputs=mobile.input, outputs=predictions)

# Train the model
# Define Top2 and Top3 Accuracy
from keras.metrics import categorical_accuracy, top_k_categorical_accuracy

def top3acc(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=3)

def top2acc(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=2)
#Loading weights(should be in the same forlder)
model.load_weights('mobile_net_final_model.h5')
# Compile the model
model.compile(Adam(lr=0.001), loss='categorical_crossentropy', metrics=[categorical_accuracy, top2acc, top3acc])
#library to read the image we can also use opencv but i do not like it
from PIL import Image
#path to the image
path=os.path.join("base_dir","val_dir","mel","ISIC_0024313.jpg")
#reading the image
raw = Image.open(path)
#display uploaded image
plt.imshow(raw)
#preprocessing the same way as in training(resizing and normalizing the image)
raw = np.array(raw.resize((224, 224)))/255.
#Make sure its RGB
raw = raw[:,:,0:3]
#Predicting the results
prediction = model.predict(np.expand_dims(raw, 0))
#print("prediction",prediction.dtype)
prediction.tolist()
Labels = ["Actinic Keratoses and intraepithelial Carcinoma","Basal Cell Carcinoma", "Benign Keratosis", "Dermatofibroma", "Melanoma","Melanocytic Nevi", "Vascular Lesions"]
#print("Labels",Labels.dtype)

#change dictionary to list of tuples and sort
results=dict(zip(Labels,prediction[0]))
#print(results)
# regular unsorted dictionary


#for k, v in results.items():
#    print(k, v)
#print(results)
# dictionary sorted by value
result_sort=OrderedDict(sorted(results.items(), key=lambda t: t[1],reverse=True))

#for k, v in result_sort.items():
#    print(k, v)

#Top 3 results
result_top3 = itertools.islice(result_sort.items(), 0, 3)

for key, value in result_top3:
    print(key,":", value)
