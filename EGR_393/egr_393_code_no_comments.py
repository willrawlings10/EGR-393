# -*- coding: utf-8 -*-

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
!pip install -U -q PyDrive
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from google.colab import auth
from oauth2client.client import GoogleCredentials
auth.authenticate_user()
gauth = GoogleAuth()
gauth.credentials = GoogleCredentials.get_application_default()
drive = GoogleDrive(gauth)


cd '/content/gdrive/My Drive/'
deer_train_path= '/content/gdrive/My Drive/Deer/deer_train'
notDeer_train_path = '/content/gdrive/My Drive/Deer/notDeer_train/'

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



deer_train = []
nDeer_train =[]
deer_train = imageAdder(deer_train_path, deer_train)
nDeer_train = imageAdder(notDeer_train_path, nDeer_train)

deer_train = np.asarray(deer_train)
nDeer_train =np.asarray(nDeer_train)

trainLabelsDeer = np.ones(len(deer_train))
trainLabelsNotDeer = np.zeros(len(nDeer_train))

x_total = np.concatenate((deer_train,nDeer_train))
y_total = np.concatenate((trainLabelsDeer, trainLabelsNotDeer))
y_total = keras.utils.np_utils.to_categorical(y_total)
y_total = y_total.astype(int)

x_train, x_test, y_train, y_test = train_test_split(x_total, y_total,test_size=0.2)

x_train = (x_train/127.5)-1
x_test = (x_test/127.5) -1

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



base_learning_rate = 0.01
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model.summary()


initial_epochs = 10
batch_for_training = 32

callbacks_list = None
history_one = model.fit(x_train, y_train, validation_data=(x_test, y_test), 
      epochs=initial_epochs, batch_size=batch_for_training)


plt.plot(history_one.history['loss'])
plt.plot(history_one.history['val_loss'])
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['training loss','validation loss'],loc='upper right')
plt.show()

keras_file = '/content/gdrive/My Drive/deer_model_new.h5'
tf.keras.models.save_model(model, keras_file)

converter = tf.lite.TFLiteConverter.from_keras_model_file(keras_file)
tflite_model = converter.convert()
open("/content/gdrive/My Drive/converted_model_nonMobileNet.tflite", "wb").write(tflite_model)

