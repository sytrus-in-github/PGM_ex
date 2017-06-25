# -*- coding: utf-8 -*-
"""
Created on Sun Jun 25 16:21:46 2017

@author: Sytrus
"""

import numpy as np
import maxflow
from PIL import Image
from matplotlib import pyplot as plt

def readImageAsGray(imageName):
    img = Image.open(imageName).convert('L')
    return np.asarray(img, dtype=np.uint8)
    
def gridGraphCut(unary, binary):
    """graph cut for 2d grid with pott-model binary cost"""
    g = maxflow.Graph[float]()
    nodeids = g.add_grid_nodes(unary.shape[:-1])
    g.add_grid_edges(nodeids, binary)
    g.add_grid_tedges(nodeids, unary[:,:,0], unary[:,:,1])
    cutValue = g.maxflow()
    isSource = g.get_grid_segments(nodeids)

    return -cutValue, isSource

if __name__=="__main__":
    img = readImageAsGray('branchAndMinCut/garden.png')
    print img.shape
    plt.imshow(img)
    plt.show()