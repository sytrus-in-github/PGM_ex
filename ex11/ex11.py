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

    # todo: check if numerically stable
    horizontal_binary = np.exp(-w * np.exp(-LAMBDA * diff_x2))
    vertical_binary = np.exp(-w * np.exp(-LAMBDA * diff_y2))

    return horizontal_binary, vertical_binary


def get_neighbour_factor(labels, horizontal_binary, vertical_binary):
    prod_neighbor_factor = np.ones(list(labels.shape).append(2))

    is_l_0 = labels[:, :-1] == 0
    is_r_0 = labels[:, 1:] == 0
    is_u_0 = labels[1:, :] == 0
    is_d_0 = labels[:-1, :] == 0

    prod_neighbor_factor[:, :-1, 0] *= np.where(is_l_0, 1, horizontal_binary)
    prod_neighbor_factor[:, :-1, 1] *= np.where(is_l_0, horizontal_binary, 1)
    prod_neighbor_factor[:, 1:, 0] *= np.where(is_r_0, 1, horizontal_binary)
    prod_neighbor_factor[:, 1:, 1] *= np.where(is_r_0, horizontal_binary, 1)
    prod_neighbor_factor[1:, :, 0] *= np.where(is_u_0, 1, vertical_binary)
    prod_neighbor_factor[1:, :, 1] *= np.where(is_u_0, vertical_binary, 1)
    prod_neighbor_factor[:-1, :, 0] *= np.where(is_d_0, 1, vertical_binary)
    prod_neighbor_factor[:-1, :, 1] *= np.where(is_d_0, vertical_binary, 1)

    return prod_neighbor_factor


def gibbs_sampling(img, unary, nb_iteration, cut_ratio, w):
    cut_value = int(cut_ratio * nb_iteration)

    num_row, num_col, _ = img.shape

    samples = np.zeros(nb_iteration - cut_value, num_row, num_col)
    horizontal_binary, vertical_binary = compute_binary(img, w)

    current_y = unary > 0.5

    for i in xrange(nb_iteration):
        factor_product = get_neighbour_factor(current_y, horizontal_binary, vertical_binary)
        factor_product = factor_product * np.stack([1 - unary, unary], -1)

        distribution = factor_product / np.sum(factor_product, axis=2)
        uniform_sample = np.random.rand(num_row, num_col)

        current_y = uniform_sample > distribution[:, :, 0]

        if i >= cut_value:
            samples[i - cut_value, :, :] = current_y

    predicted_labels = np.mean(samples, axis=0)
    return predicted_labels


if __name__ == '__main__':
    img = read_unary('in2329-supplementary_material_11/5_18_s_dict.yml')
    plt.imshow(img)
    plt.show()
