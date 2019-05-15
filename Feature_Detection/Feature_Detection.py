# coding: utf-8

# In[6]:


import os
import sys

sys.path.append(os.pardir)
from Notebooks.LinkDatabases.ImageDescriptors import DescriptorsDatabase

descriptorsDb = DescriptorsDatabase()

# In[7]:


import os
from imageio import imwrite
import cv2

pickle_file = "images.pkl"
base = "./denoised_images"
new_base = "./features"


def getImageList():
    files = []
    for file in os.listdir(new_base):
        files.append(file)
    return files


def writeFile(new_base, file):
    try:
        imwrite(os.path.join(new_base, file), img)
    except:
        print("Error writing the file: ", file)


l = []
files = []  # getImageList()
for file in os.listdir(base):
    if not file in files:
        if ".png" in file or ".jpg" in file:
            full_path = os.path.join(base, file)
            img = cv2.imread(full_path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            orb = cv2.ORB_create()
            kp = orb.detect(gray, None)
            kp, des = orb.compute(gray, kp)
            l.append(des)
            des_str = str(des.tolist())
            descriptorsDb.insertDescriptorData(file[:-4], des_str)

            # Draw keypoints and write the photo
            # img=cv2.drawKeypoints(gray,kp,cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            # writeFile(new_base,file)

# In[10]:


kp
