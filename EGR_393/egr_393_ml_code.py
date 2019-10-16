# -*- coding: utf-8 -*-
"""EGR_393_ML_Code.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1yzXHOq4OvBMrbdIz9wJOeTFvAL220Vvz

*Import the needed tensorflow and python libraries needed for data modification*
"""

from __future__ import absolute_import, division, print_function, unicode_literals


import tensorflow as tf

import cv2
import PIL
import imageio
import os
import glob
import csv
import math
from pathlib import Path
import numpy as np
from skimage import io
from skimage import transform
from skimage import draw
from skimage import exposure
import matplotlib.pyplot as plt
import pickle
import requests
import tarfile
import dlib
import sys
from PIL import Image
from PIL.ExifTags import TAGS
!pip install -q tf-nightly
from sklearn.model_selection import train_test_split

import tarfile
import keras
import errno
import urllib
try:
    from imageio import imsave
except:
    from scipy.misc import imsave

  
import sys
import urllib.request
from urllib.request import urlretrieve
from csv import reader
import os.path
from PIL import Image
import requests
from io import BytesIO
from skimage import io
import matplotlib.image as mpimg
from scipy import ndimage, misc
import h5py
from IPython.display import display
from IPython.display import Image as _Imgdis
from PIL import Image
from scipy import ndimage

"""*Import the Google Drive Authorization needed to access the data folders*"""

from google.colab import drive
drive.mount('/content/gdrive')
!pip install -U -q PyDrive ## you will have install for every colab session
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from google.colab import auth
from oauth2client.client import GoogleCredentials
# 1. Authenticate and create the PyDrive client.
auth.authenticate_user()
gauth = GoogleAuth()
gauth.credentials = GoogleCredentials.get_application_default()
drive = GoogleDrive(gauth)

"""*Move into My Drive, where my image folders are stored*"""

cd '/content/gdrive/My Drive/'

"""*Create the paths to the directories of the images, which are stored in their own folders in My Drive*"""

deer_train_path= '/content/gdrive/My Drive/Deer/deer_train'
notDeer_train_path = '/content/gdrive/My Drive/Deer/notDeer_train/'

"""@param: path_in - *The path to the folder of images*
@param: arrayName *An array to store the images*

Every file (or image) in the folder is read in RGB format, as BGR is the default for openCV

The image is the resized to (224,224), which is the height and width ised by most pre-trained models

The image is then appended to an array

The if statement is included because one of the images in the not deer training folder, at index 465, was not working and this bypassed that image 

@return: ArrayName *The same images now populated with image data*
"""

def imageAdder(path_in, arrayName):
   
    path = path_in
    i=0
    for image_path in os.listdir(path):
            input_path = os.path.join(path, image_path)
            image = cv2.imread(input_path)
            image2 =cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            #image = (image/127.5) - 1
            img = cv2.resize(image2,(224,224)) #crop rather than resize -keeps aspect ratio
            arrayName.append(img)
            print(image_path)
            i = i+1
    return arrayName

"""*Instantiating the arrays that will hold the training and testing images for deer and not deer*"""

deer_train = []
nDeer_train =[]

"""*Populating the previously instantiated arrays with image data from the folders*""""

deer_train = imageAdder(deer_train_path, deer_train)

nDeer_train = imageAdder(notDeer_train_path, nDeer_train)

"""*Converting the arrays from lists to NumPy arrays, which allows for multi-dimensionality and is how the data will be fed into the model*"""

deer_train = np.asarray(deer_train)
nDeer_train =np.asarray(nDeer_train)


"""*Creating the label arrays for the images. Assigning 1 to indicate a deer and 0 to indicate a not deer. Using np.ones or np.zeros created a numpy array of 1's or 0's, which have their length determined by the length of the corresponding image arrays. The prints after the assignments are used to test whether the labels were generated correctly.*"""

trainLabelsDeer = np.ones(len(deer_train))
trainLabelsNotDeer = np.zeros(len(nDeer_train))

"""*This adds the training labels together and the testing labels together in one array. The arrays are in order, so the first half are deer images and the second half are not deer images.*"""

x_total = np.concatenate((deer_train,nDeer_train))
y_total = np.concatenate((trainLabelsDeer, trainLabelsNotDeer))
y_total = keras.utils.np_utils.to_categorical(y_total)
y_total = y_total.astype(int)

x_train, x_test, y_train, y_test = train_test_split(x_total, y_total,test_size=0.2)



"""*Normalizing the images around a mean offset of 0 to put them into the model*"""

x_train = (x_train/127.5)-1
x_test = (x_test/127.5) -1



"""*Importing the libararies needed to create the CNN architecture*"""

import keras
from keras.layers import *
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.models import save_model
from keras.models import load_model
from tensorflow.keras import backend as K
from keras.layers.core import Dense, Activation
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.models import Model
from keras.applications import imagenet_utils
from keras.layers import Dense,GlobalAveragePooling2D
from keras.applications.mobilenet import preprocess_input
from IPython.display import Image
from keras.optimizers import Adam



"""*Specifiying how many classes there are, setting the basis of our model as MobileNetV2 (which is trained on imageNet, does not have the last classifcation layer, and with images of size 224x224x3). This allows our model to be based of an already trained ML model. A sequential model is created, with ResNet50 as the base. A flatten() is added so the images are in the right dimension for the softmax function. This last layer allows us to "predict" what class the image is labelled as based on the probability of the softmax function*"""

IMG_SHAPE = (224, 224, 3)

base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                               include_top=False,
                                               weights='imagenet')

base_model.trainable = False

global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
prediction_layer = tf.keras.layers.Dense(2, activation = "softmax")
    
model = tf.keras.Sequential([
  base_model
])

model.add(global_average_layer)
model.add(prediction_layer)

"""Compile the model"""

base_learning_rate = 0.01
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model.summary()

"""*Training the model with 10 epochs at 32 images per batch. This means that the model will run through all the data 10 times in segments of 32 images at a time, with x_train being the training set and y_test being the validation set.*"""

initial_epochs = 10
batch_for_training = 32


callbacks_list = None
history_one = model.fit(x_train, y_train, validation_data=(x_test, y_test), 
      epochs=initial_epochs, batch_size=batch_for_training)

"""*Showing a graph of the training loss and the validation loss on a loss vs epoch graph*"""

plt.plot(history_one.history['loss'])
plt.plot(history_one.history['val_loss'])
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['training loss','validation loss'],loc='upper right')
plt.show()

"""*Saving the model then coverting the Tensorflow Model just trained into a Tensorflow Lite Model. This allows the model to be run on mobile devices.*"""

keras_file = '/content/gdrive/My Drive/deer_model_new.h5'
tf.keras.models.save_model(model, keras_file)

# Convert to TensorFlow Lite model.
converter = tf.lite.TFLiteConverter.from_keras_model_file(keras_file)
tflite_model = converter.convert()
open("/content/gdrive/My Drive/converted_model_nonMobileNet.tflite", "wb").write(tflite_model)

