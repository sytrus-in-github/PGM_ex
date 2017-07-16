import numpy as np
import matplotlib.pyplot as plt
import utils
from PIL import Image

try:
    import cPickle as pickle
except:
    import pickle

if __name__ == '__main__':

    # image = Image.open('in2329-supplementary_material_10/12_33_s.bmp').convert('L')
    # plt.imshow(image)
    # plt.show()



    data, unary_energies = utils.read_unary('12_33_s_unary.txt')
    unary_labeling = np.argmax(data, axis=2) * 10
    unary_labeling = np.transpose(unary_labeling, (1, 0))
    plt.imshow(unary_labeling)
    plt.draw()


    # plt.imshow(unary_labeling)
    # plt.show()
    #
    with open('in2329-supplementary_material_10/12_33_s_segmentation', 'r') as filecontent:
       q = pickle.load(filecontent)
       q = np.transpose(q, (1, 0, 2))

       labeling = np.argmax(q, axis=2) * 10
       plt.figure()
       plt.imshow(labeling)
       plt.show()