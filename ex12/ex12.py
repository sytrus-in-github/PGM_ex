import numpy as np
import os
import yaml
from matplotlib import pyplot as plt
from PIL import Image
import maxflow

NP_DTYPE_MAP = {'f': np.float32, 'd': np.float64}
LAMBDA = 0.5


def cvDat2nparray(cvdict):
    return np.array(cvdict['data'],
                    dtype=NP_DTYPE_MAP[cvdict['dt']]).reshape(cvdict['rows'],
                                                              cvdict['cols'])


def read_unary(yml_file):
    with open(yml_file) as filecontent:
        string = filecontent.read()
        string = "%YAML 1.0" + os.linesep + "---" + string[len("%YAML:1.0"):] if string.startswith(
            "%YAML:1.0") else string
        cvdict = yaml.load(string)['unary']
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
    l = list(labels.shape)
    prod_neighbor_factor = np.ones(l + [2])

    is_l_0 = labels[:, :-1] == 0
    is_r_0 = labels[:, 1:] == 0
    is_d_0 = labels[1:, :] == 0
    is_u_0 = labels[:-1, :] == 0

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

    samples = np.zeros((nb_iteration - cut_value, num_row, num_col), dtype=np.float64)
    horizontal_binary, vertical_binary = compute_binary(img, w)

    current_y = unary

    for i in xrange(nb_iteration):
        factor_product = get_neighbour_factor(current_y, horizontal_binary, vertical_binary)
        factor_product = factor_product * np.stack([1 - unary, unary], -1)

        distribution = factor_product / np.expand_dims(np.sum(factor_product, axis=2), 2)
        uniform_sample = np.random.rand(num_row, num_col)

        current_y = uniform_sample > distribution[:, :, 0]

        if i >= cut_value:
            samples[i - cut_value, :, :] = current_y

    predicted_labels = np.mean(samples, axis=0)
    return predicted_labels


def gridGraphCut(unary, horizontal_binary, vertical_binary):
    """graph cut for 2d grid with pott-model binary cost"""
    nbrow, nbcol = unary.shape
    g = maxflow.Graph[float]()
    nodeids = g.add_grid_nodes((nbrow, nbcol))
    #g.add_grid_edges(nodeids, binary)
    g.add_grid_tedges(nodeids, unary[:,:,0], unary[:,:,1])
    cutValue = g.maxflow()
    isSource = g.get_grid_segments(nodeids)

    return cutValue, isSource


if __name__ == '__main__':
    train_directory = 'data/cows-training'
    unary_directory = 'data/cows-unary'

    image_filenames = [fn for fn in os.listdir(train_directory) if fn.endswith('.bmp')]
    unaries_filenames = [fn for fn in os.listdir(unary_directory) if fn.endswith('.yml')]

    for img_name, unary_name in zip(image_filenames, unaries_filenames):
        img = Image.open(os.path.join(train_directory, img_name)).convert('RGB')
        unaries = read_unary(os.path.join(unary_directory, unary_name))

        prediction = gibbs_sampling(img, unaries, 2000, 0.8, 4.2)
        result = (prediction > 0.5) * 255

        result_image_name = os.path.join('cows-groundtruth', img_name)
        Image.fromarray(result).save(result_image_name)

    # img_unaries = read_unary('in2329-supplementary_material_11/5_21_s_dict.yml')
    #
    # img = Image.open('in2329-supplementary_material_11/5_21_s.bmp').convert('RGB')
    #
    # img = np.asarray(img, dtype=np.float64) / 255.
    # prediction = gibbs_sampling(img, img_unaries, 2000, 0.8, 4.2)
    #
    # result = (prediction > 0.5) * 255
    #
    # plt.imshow((img_unaries > 0.5) * 255)
    # plt.draw()
    #
    # plt.figure()
    # plt.imshow(result)
    # plt.show()
