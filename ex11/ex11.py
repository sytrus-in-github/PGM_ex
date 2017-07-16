import numpy as np
import yaml
from matplotlib import pyplot as plt

NP_DTYPE_MAP = {'f': np.float32, 'd': np.float64}
LAMBDA = 0.5

def cvDat2nparray(cvdict):
    return np.array(cvdict['data'],
                    dtype=NP_DTYPE_MAP[cvdict['dt']]).reshape(cvdict['rows'],
                                                              cvdict['cols'])


def read_unary(yml_file):
    with open(yml_file) as filecontent:
        cvdict = yaml.load(filecontent)['unary']
        print cvdict.keys()
        return cvDat2nparray(cvdict)
        

def compute_binary(img, w):
    diff_x2 = (img[:, 1:, :] - img[:, :-1, :]) ** 2
    diff_y2 = (img[1:, :, :] - img[:-1, :, :]) ** 2

    horizontal_binary = np.exp(-w * np.exp(-LAMBDA * diff_x2))
    vertical_binary = np.exp(-w * np.exp(-LAMBDA * diff_y2))

    return horizontal_binary, vertical_binary


def get_neighbour_factor(coordinate, horizontal_binary, vertical_binary):
    r, c = coordinate
    rows = horizontal_binary.shape[0]
    cols = vertical_binary.shape[1]
    factors = []
    if r > 0:
        factors.append(vertical_binary[r-1, c])
    if r < rows - 1:
        factors.append(vertical_binary[r, c])
    if c > 0:
        factors.append(horizontal_binary[r, c-1])
    if c < cols - 1:
        factors.append(horizontal_binary[r, c])
    return factors


def gibbs_sampling(img, unary, nb_iteration, cut_ratio, w):
    cut_value = int(cut_ratio * nb_iteration)
    num_row, num_col, _ = img.shape
    samples = np.zeros()

    for i in xrange(nb_iteration):

    return predicted_labels


if __name__ == '__main__':
    img = read_unary('in2329-supplementary_material_11/5_18_s_dict.yml')
    plt.imshow(img)
    plt.show()
