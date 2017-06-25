# -*- coding: utf-8 -*-
"""
Created on Sun Jun 25 16:21:46 2017

@author: Sytrus
"""

import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

def readImageAsGray(imageName):
    img = Image.open(imageName).convert('L')
    return np.asarray(img, dtype=np.uint8)
    
if __name__=="__main__":
    img = readImageAsGray('branchAndMinCut/garden.png')
    print img.shape
    plt.imshow(img)
    plt.show()