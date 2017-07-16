import numpy as np
import yaml
from matplotlib import pyplot as plt


NP_DTYPE_MAP = {'f': np.float32, 'd':np.float64}


def cvDat2nparray(cvdict):
    return np.array(cvdict['data'], 
                    dtype=NP_DTYPE_MAP[cvdict['dt']]).reshape(cvdict['rows'], 
                                                              cvdict['cols'])


def read_unary(yml_file):
    with open(yml_file) as filecontent:
        cvdict = yaml.load(filecontent)['unary']
        print cvdict.keys()
        return cvDat2nparray(cvdict)
        


if __name__ == '__main__':
    img = read_unary('in2329-supplementary_material_11/5_18_s_dict.yml')
    plt.imshow(img)
    plt.show()