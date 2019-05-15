# coding: utf-8

# In[9]:

import os
import sys

sys.path.append(os.pardir)
from Notebooks.LinkDatabases.FacebookData import FacebookDataDatabase
from imageio import imwrite
import cv2
import urllib.request
from PIL import Image
import re
import os
import numpy as np
from numpy import array
from scipy.ndimage import filters
import ntpath

facebookDb = FacebookDataDatabase()
image_regex = "([A-Za-z_\d]+.jpg|[A-Za-z_\d]+.png)"


def getExistingImages():
    files = []
    base = "./images"
    for file in os.listdir(base):
        files.append(file)
    return files


def denoise_image(full_path):
    im = Image.open(full_path)
    if im.size == (1, 1,):
        os.remove(full_path)
        return
    im.resize((360, 360))
    im.convert('L')
    im_gauss = filters.gaussian_filter(im, 5)
    kernel = np.ones((5, 5), np.uint8)
    dilation = cv2.dilate(im_gauss, kernel, iterations=1)
    erosion = cv2.erode(dilation, kernel, iterations=1)
    try:
        denoised = cv2.fastNlMeansDenoisingColored(erosion, None, 10, 10, 7, 21)
        flat = array(denoised)
    except:
        print("Error denoising file: ", full_path)
    try:
        imwrite(full_path, flat)
    except:
        print("Error writing the file: ", full_path)


# mean, eigenVectors = cv2.PCACompute(data, mean=None, maxComponents=NUM_EIGEN_FACES)
# averageFace = mean.reshape(sz)

from Notebooks.LinkDatabases.ImageDescriptors import DescriptorsDatabase

descriptorsDb = DescriptorsDatabase()
import multiprocessing as mp

queue = mp.Queue()


def get_features(full_path):
    file = ntpath.basename(full_path)
    img = cv2.imread(full_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BayerBG2GRAY)
    orb = cv2.ORB_create()
    kp = orb.detect(gray, None)
    kp, des = orb.compute(gray, kp)
    des_str = str(des.tolist())
    descriptorsDb.insertDescriptorData(file[:-4], des_str)


class ImagePathData:
    def __init__(self, url, path):
        self.url = url
        self.path = path


shared_images = []
files = getExistingImages()
print(len(facebookDb.selectFacebookData()))
for row in facebookDb.selectFacebookData():
    url = row[1]
    image_matches = re.findall(image_regex, url)
    if image_matches:
        image_name = str(row[0]) + image_matches[0][-4:]
        if not image_name in files:
            path = "./images/" + image_name
            imageData = ImagePathData(url, path)
            shared_images.append(imageData)
    else:
        print("NO!: ", url)


def download(images):
    for imageData in images:
        url = imageData.url
        path = imageData.path
        try:
            urllib.request.urlretrieve(url, path)
            denoise_image(path)
        except:
            print("download failed for {0}".format(url))


def group(iterator, count):
    itr = iter(iterator)
    while True:
        yield tuple([next(itr) for i in range(count)])


import threading

for image_group in group(shared_images, 800):
    threads = []
    for images in group(image_group, int(len(image_group) / 8)):
        print(len(images))
        t = threading.Thread(target=download, args=[images])
        t.start()
        threads.append(t)
    for thread in threads:
        thread.join()
