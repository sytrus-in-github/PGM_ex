import numpy as np
import os
import warnings
import yaml
from matplotlib import pyplot as plt
from PIL import Image
import maxflow

NP_DTYPE_MAP = {'f': np.float32, 'd': np.float64}
LAMBDA = 0.5


def convertBadOpenCVYAMLString(string):
    bad_head = "%YAML:1.0\nunary: !!opencv-matrix"
    good_head = "%YAML 1.0\n---\nunary: !!map"
    if string.startswith(bad_head):
        string = good_head + string[len(bad_head):]
    else:
        print 'not changed.'
    return string


def cvDat2nparray(cvdict):
    return np.array(cvdict['data'],
                    dtype=NP_DTYPE_MAP[cvdict['dt']]).reshape(cvdict['rows'],
                                                              cvdict['cols'])


def read_unary(yml_file=None, npy_file=None):
    if npy_file is not None:
        if os.path.isfile(npy_file):
            print 'Loading', npy_file
            return np.load(npy_file)
    with open(yml_file) as filecontent:
        string = filecontent.read()
        string = convertBadOpenCVYAMLString(string)
        cvdict = yaml.load(string)['unary']
        dat = cvDat2nparray(cvdict)
    if npy_file is not None:
        print 'Saving', npy_file
        np.save(npy_file, dat)
    return dat


def squared_norm(p1, p2):
    return np.sum((np.array(p1) - np.array(p2)) ** 2, axis=2)


def compute_energy(img):
    diff_x2 = squared_norm(img[:, 1:, :], img[:, :-1, :])
    diff_y2 = squared_norm(img[1:, :, :], img[:-1, :, :])

    horizontal_binary = np.exp(-LAMBDA * diff_x2)
    vertical_binary = np.exp(-LAMBDA * diff_y2)

    return horizontal_binary, vertical_binary


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
    return predicted_labels, samples


def gridGraphCut(unary, horizontal_binary, vertical_binary):
    """graph cut for 2d grid with pott-model binary cost"""
    nbrow, nbcol, _ = unary.shape
    g = maxflow.Graph[float]()
    nodeids = g.add_grid_nodes((nbrow, nbcol))
    g.add_grid_tedges(nodeids, unary[:, :, 0], unary[:, :, 1])
    #    # add horizontal binary weights
    #    for r in xrange(nbrow):
    #        for c in xrange(nbcol-1):
    #            energy = horizontal_binary[r,c]
    #            g.add_edge(nodeids[r,c], nodeids[r, c+1], energy, energy)
    #    # add vertical binary weights
    #    for r in xrange(nbrow-1):
    #        for c in xrange(nbcol):
    #            energy = vertical_binary[r,c]
    #            g.add_edge(nodeids[r,c], nodeids[r+1, c], energy, energy)
    structure = np.zeros((3, 3))
    structure[1, 2] = 1
    g.add_grid_edges(nodeids[:, :-1], structure=structure, weights=horizontal_binary, symmetric=True)
    structure = np.zeros((3, 3))
    structure[2, 1] = 1
    g.add_grid_edges(nodeids[:-1, :], structure=structure, weights=vertical_binary, symmetric=True)
    cutValue = g.maxflow()
    isSource = g.get_grid_segments(nodeids)

    return cutValue, np.logical_not(isSource)


def getMasks(labeling):
    horizontal_mask = labeling[:, :-1] != labeling[:, 1:]
    vertical_mask = labeling[:-1, :] != labeling[1:, :]
    return horizontal_mask, vertical_mask


def lossMinimizingParameterLearning(imgs, gts, unaries, T, C):
    """Subgradient descent S-SVM learning. See page 26 of lecture slides 11.
    imgs: list of images as float ndarray
    gts: list of groundtruths as boolean ndarray (0 background / 1 foreground)
    unaries: list of unary eneries as float ndarray (not factors !!!)
    T: number of iterations as int
    C: regularizer as number"""
    N = len(imgs)
    C = float(C)
    w = 0.
    binaries = [compute_energy(img) for img in imgs]
    imgs = None  # release useless data from memory
    # get ground-truth horizontal/vertical masks for binary energy
    gt_masks = [getMasks(gt) for gt in gts]
    # get shape of horizontal/vertical binary shapes
    bh_shape = binaries[0][0].shape
    bv_shape = binaries[0][1].shape

    for t in xrange(1, T + 1):
        # to store the sum of vn as horizontal/vertical matrices
        vn_h = np.zeros(bh_shape)
        vn_v = np.zeros(bv_shape)

        for n in xrange(N):
            yn = gts[n]
            nb_pixel = len(yn.flatten())
            unary = unaries[n].copy()
            unary = np.stack([1 - unary, unary], -1)
            binary_h, binary_v = binaries[n]
            gt_mask_h, gt_mask_v = gt_masks[n]
            # integrate hamming loss to unary
            unary[:, :, 0] += np.where(yn, 0., 1. / nb_pixel)
            unary[:, :, 1] += np.where(yn, 1. / nb_pixel, 0.)
            _, y_ = gridGraphCut(unary, w * binary_h, w * binary_v)
            mask_h, mask_v = getMasks(y_)
            vn_h += (gt_mask_h * binary_h - mask_h * binary_h)
            vn_v += (gt_mask_v * binary_v - mask_v * binary_v)

        # reshape w as matrices for update
        w_h = w * np.ones(bh_shape)
        w_v = w * np.ones(bv_shape)
        print np.mean(vn_h), np.mean(vn_v)
        w_h -= (1. / t) * (w_h + (C / N) * vn_h)
        w_v -= (1. / t) * (w_v + (C / N) * vn_v)
        print np.mean(w_h), np.mean(w_v), np.mean(np.concatenate([w_h.flatten(), w_v.flatten()]))
        # get scalar w back as mean value
        w = np.mean(np.concatenate([w_h.flatten(), w_v.flatten()]))
        print 'iteration', t, 'w', w
    return w


def probabilisticParameterLearning(imgs, gts, unaries, T):
    N = len(imgs)
    w = 0.
    lambda_param = 0.01

    binaries = [compute_energy(img) for img in imgs]
    gt_masks = [getMasks(gt) for gt in gts]

    bh_shape = binaries[0][0].shape
    bv_shape = binaries[0][1].shape

    for t in xrange(1, T + 1):
        diff_h = np.zeros(bh_shape)
        diff_v = np.zeros(bv_shape)

        for n in xrange(N):
            unary = unaries[n]
            img = imgs[n]

            binary_h, binary_v = binaries[n]
            gt_mask_h, gt_mask_v = gt_masks[n]

            _, samples = gibbs_sampling(img, unary, 500, 0.8, w)

            masks_for_samples = map(list, zip(*[getMasks(sample) for sample in samples]))

            masks_h = np.asarray(masks_for_samples[0])
            masks_v = np.asarray(masks_for_samples[1])

            expected_energies_h = np.mean(np.expand_dims(binary_h, 0) * masks_h, axis=0)
            expected_energies_v = np.mean(np.expand_dims(binary_v, 0) * masks_v, axis=0)

            diff_h += gt_mask_h * binary_h - expected_energies_h
            diff_v += gt_mask_v * binary_v - expected_energies_v

        w_h = w * np.ones(bh_shape)
        w_v = w * np.ones(bv_shape)

        w_h -= (1. / t) * (2 * w_h * lambda_param + diff_h)
        w_v -= (1. / t) * (2 * w_v * lambda_param + diff_v)

        w = np.mean(np.concatenate([w_h.flatten(), w_v.flatten()]))
        print 'iteration', t, 'w', w

    return w


def test_probabilisticParameterLearning():
    gts, imgs, unaries = setup_data_for_training()
    T = 10

    print 'learning w ...'
    probabilisticParameterLearning(imgs, gts, unaries, T)


def test_lossMinimizingParameterLearning():
    gts, imgs, unaries = setup_data_for_training()
    T = 10
    C = 100

    print 'learning w ...'
    lossMinimizingParameterLearning(imgs, gts, unaries, T, C)


def setup_data_for_training():
    print 'indexing images ...'
    train_directory = 'data/cows-training'
    unary_directory = 'data/cows-unary'
    truth_directory = 'data/cows-groundtruth'
    filenames = [fn[:-4] for fn in os.listdir(train_directory) if fn.endswith('.bmp')]

    print 'loading images ...'
    imgs = [np.asarray(
        Image.open(
            os.path.join(train_directory, f + '.bmp')
        ).convert('RGB'),
        dtype=np.float64
    ) / 255. for f in filenames]
    print 'loading ground-truth segmentations ...'

    gts = [np.asarray(
        Image.open(
            os.path.join(truth_directory, f + '.bmp')
        ),
        dtype=np.uint8
    ) > 127 for f in filenames]
    print 'loading unary energies ...'

    warnings.simplefilter("ignore")

    unaries = [-np.log(read_unary(os.path.join(unary_directory, f + '.yml'), os.path.join(unary_directory, f + '.npy')))
               for f in filenames]
    warnings.resetwarnings()

    return gts, imgs, unaries


if __name__ == '__main__':
    test_lossMinimizingParameterLearning()
    #test_probabilisticParameterLearning()
    raise Exception('stop.')
    train_directory = 'data/cows-training'
    unary_directory = 'data/cows-unary'

    image_filenames = [fn for fn in os.listdir(train_directory) if fn.endswith('.bmp')]
    unaries_filenames = [fn for fn in os.listdir(unary_directory) if fn.endswith('.yml')]

    for img_name, unary_name in zip(image_filenames, unaries_filenames):
        img = Image.open(os.path.join(train_directory, img_name)).convert('RGB')
        unaries = read_unary(os.path.join(unary_directory, unary_name))

        img = np.asarray(img, dtype=np.float64) / 255.

        print 'Computing ground truth for image: ', img_name
        prediction = gibbs_sampling(img, unaries, 2000, 0.8, 4.2)
        result = (prediction > 0.5) * 255
        result = result.astype(np.uint8)

        result_image_name = os.path.join('data/cows-groundtruth', img_name)
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
