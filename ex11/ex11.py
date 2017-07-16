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


def squared_norm(p1, p2):
    return np.sum((np.array(p1) - np.array(p2)) ** 2, axis=2)


def compute_binary(img, w):
    diff_x2 = squared_norm(img[:, 1:, :], img[:, :-1, :])
    diff_y2 = squared_norm(img[1:, :, :], img[:-1, :, :])

    #todo: check if numerically stable
    horizontal_binary = np.exp(-w * np.exp(-LAMBDA * diff_x2))
    vertical_binary = np.exp(-w * np.exp(-LAMBDA * diff_y2))

    return horizontal_binary, vertical_binary


def get_neighbour_factor(labels, horizontal_binary, vertical_binary):
    rows = horizontal_binary.shape[0]
    cols = vertical_binary.shape[1]
#    horizontal_padded = np.pad(horizontal_binary, ((0,0),(1,1)), 'constant', constant_values = 1)
#    vertical_padded = np.pad(vertical_binary, ((1,1),(0,0)), 'constant', constant_values = 1)
    prod_neighbor_factor = np.ones(list(labels.shape).append(2))
    
    
    return prod_neighbor_factor


def gibbs_sampling(img, unary, nb_iteration, cut_ratio, w):
    cut_value = int(cut_ratio * nb_iteration)

    num_row, num_col, _ = img.shape

    samples = np.zeros(nb_iteration - cut_value, num_row, num_col)
    horizontal_binary, vertical_binary = compute_binary(img, w)

    current_y = unary

    for i in xrange(nb_iteration):

        

    return predicted_labels


if __name__ == '__main__':
    img = read_unary('in2329-supplementary_material_11/5_18_s_dict.yml')
    plt.imshow(img)
    plt.show()
