# -*- coding: utf-8 -*-


from __future__ import absolute_import, division, print_function, unicode_literals

!pip install tensorflow==1.14.0rc1

import tensorflow as tf

import cv2
import imageio
import os
import csv
from csv import reader
import math
from pathlib import Path
import numpy as np
from skimage import io
from io import BytesIO
from skimage import transform
from skimage import draw
from skimage import exposure

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
import requests
import sys
import PIL
from PIL import Image
from PIL.ExifTags import TAGS
!pip install -q tf-nightly

import errno
import urllib 
import urllib.request
from urllib.request import urlretrieve
import os.path
import requests

import h5py

"""Import the Google Drive Authorization needed to access the data folders"""

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



cd '/content/gdrive/My Drive/'
!ls '/content/gdrive/My Drive/'

"""*  @param: fileName - *Name of .csv file with the image urls*
*  @param: folderName *Name of the folder the images will be added to *
*  @param: pictName - *The base name assigned to the images*
*  @param: arrayName *The name of the array that the images will be added to *
*  @param: first - *Index of where the image addition should start*
*  @param: last - *index of where the image addition should stop*

This method opens a .csv file, loops through the images, and adds the images to a folder where they will stored as well as an array to store the data, if you wanted to pickle that array to your drive so as not to loop through the images folder.

The url is retrieved, the image is red and appended to the folder with the extension .jpg and the image array data is appended to arrayName.
"""

def canOpener(fileName, folderName, pictName,arrayName, first, last,):
  #only do once, as they will now be in the folders from the .csv files
  csv_filename = fileName
  placement = folderName
  a= first
  b = last
  j=0
  i=pictName + str(j)
  with open(csv_filename+".csv".format(csv_filename), 'r') as csv_file:
    for line in reader(csv_file):
      if  a<=j<=b and line[0] != '':
        urlretrieve(line[0], placement + i + ".jpg")
        img=io.imread(line[0])
        arrayName.append(img)
        print(j)
        j=j+1
        i=pictName + str(j)
        j=int(j)
               
      else:
        j=j+1

"""*  @param: rgb - *Image in rgb format*

This method takes an image in rbg format and turns it into a grayscale image with only one channel.

* @return: *Images in grayscale format*
"""

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

    return gray

"""*  @param: arrayOld - *Array containing images*
*  @param: arrayNew *Array where the resized images *

This method resizes the images according to the WIDTH and HEIGHT instantiated earlier in the code. These resized images are then appended to a new array.
"""

def imageShaper(arrayOld,arrayNew):
  for img in arrayOld:
      full_size_image = img
      imageUp = cv2.resize(full_size_image, (WIDTH,HEIGHT), interpolation=cv2.INTER_CUBIC)
      #imageIn = rgb2gray(imageUp)
      arrayNew.append(imageIn)


np_load_old = np.load
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)


deer_train = []
nDeer_train =[]

deer_test =[]
nDeer_test =[]



canOpener('deer',"deer_train/", "Deer_Train", deer_train, 0,1500)
canOpener('deer',"deer_test/", "Deer_Test", deer_test, 1501,1900)


canOpener('notDeer',"notDeer_train/", "Not_Deer_Train", nDeer_train, 0,1500)
canOpener('notDeer',"notDeer_test/", "Not_Deer_Test", nDeer_test, 1501,1899)

**Commented Out**

These arrays are stored for future use. However, for the training of the deer model, I found it more useful to just loop through the folders, as it helped me retain RAM.
"""
np.save("deer_train", deer_train)
np.save("deer_test", deer_test)

np.save("nDeer_train", nDeer_train)
np.save("nDeer_test", nDeer_test)

**Commented Out**

The old arrays are loaded, the images resized, and these new images appended to another array. These arrays are then saved to My Drive.

deer_train = np.load('deer_train.npy')
imageShaper(deer_train, deer_train_fixed)
np.save("deer_train_fixed", deer_train_fixed)

nDeer_train = np.load('nDeer_train.npy')
imageShaper(nDeer_train, nDeer_train_fixed)
np.save("nDeer_train_fixed", nDeer_train_fixed)

deer_test = np.load('deer_test.npy')
imageShaper(deer_test, deer_test_fixed)
np.save("deer_test_fixed", deer_test_fixed)

nDeer_test = np.load('nDeer_test.npy')
imageShaper(nDeer_test, nDeer_test_fixed)
np.save("nDeer_test_fixed", nDeer_test_fixed)

**Commented Out**

This is a little way to loop through the images and make sure they are all standardized and reshaped correctly. The k is used as a tracker for me to see how far the program has progressed.

k=0

for i in nDeer_train_fixed:
  print(k, i.shape)
  k=k+1
"""
