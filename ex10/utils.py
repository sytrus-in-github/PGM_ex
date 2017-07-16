import numpy as np
from PIL import Image

try:
    import cPickle as pickle
except:
    import pickle

coeff = {
    'w1': 10,
    'w2': 3,
    'theta_alpha': 80,
    'theta_beta': 13,
    'theta_gamma': 3,
}


def map_energy(unary):
    if unary == 0:
        return float("inf")
    return -np.log(unary)


def squared_norm(p1, p2):
    return np.sum((np.array(p1) - np.array(p2)) ** 2, axis=0)


def diagnosis(q_old, q_new, normalizer):
    col, row, klass = q_old.shape
    problems = []
    for c in xrange(col):
        for r in xrange(row):
            if normalizer[c, r] == 0:
                print c, r
                print q_old[c, r, :]
                print q_new[c, r, :]
                problems.append((c, r, q_old[c, r, :], q_new[c, r, :]))
    return problems


def update_q(q_old, unary_energy, binary_energy):
    col, row, klass = q_old.shape
    labels = np.argmax(q_old, axis=-1)
    is_valid = np.max(q_old, axis=-1) != 1.
    q_new = np.zeros(q_old.shape)
    # update q_new
    for c in xrange(col):
        # print c
        for r in xrange(row):
            if not is_valid[c, r]:
                q_new[c, r, :] = q_old[c, r, :]
                continue
            label_i = labels[c, r]
            message_i = (np.expand_dims(binary_energy[r, c, :, :].T * (labels != label_i), 2) * q_old).reshape(-1, klass)

            un_en = unary_energy[c, r, :]
#            un_en_mask = np.isinf(un_en)
#
            np_sum = np.sum(message_i, axis=0)
#            np_sum_mask = np.isinf(np_sum)

            # print np.max(un_en * (1 - un_en)), np.max(np_sum)
            exponent = -un_en - np_sum
            exponent -= np.max(exponent)
            energy_update = np.exp(exponent)

            q_new[c, r, :] = energy_update

            # if np.sum(energy_update) != 0:
            #     q_new[c, r, :] = energy_update
            # else:
            #     q_new[c, r, :] = q_old[c, r, :]

    # normalize q_new to have 1 sum
    normalizer = np.expand_dims(np.sum(q_new, axis=-1), 2)

    problems = diagnosis(q_old, q_new, normalizer)

    print 'problem entries number:', len(problems)

    q_new /= normalizer

    return q_new


def read_unary(filename):
    with open(filename, 'r') as filecontent:
        nbcol, nbrow, nbclass = [int(s) for s in filecontent.readline().split()]
        data_list = [float(s) for s in filecontent.readline().split()]

        energies = np.array(map(lambda unary: map_energy(unary), data_list)).reshape((nbcol, nbrow, nbclass))

        data = np.array(data_list).reshape((nbcol, nbrow, nbclass))
    return data, energies


def precompute_binary_map(image, image_name):
    filename = 'in2329-supplementary_material_10/' + image_name + '.memmap'

    row, col, _ = image.shape
    binary_map = np.memmap(filename, dtype=np.float64, mode='w+', shape=(row, col, row, col))

    image_stacked = np.reshape(image, (-1, 3)).T
    grid = np.mgrid[0:row, 0:col].reshape(2, -1)

    for r in xrange(row):
        print r
        for c in xrange(col):
            points_square_norm = squared_norm(grid, [[r], [c]])

            binary_energy = coeff['w1'] * np.exp(-points_square_norm / (2 * (coeff['theta_alpha'] ** 2)) -
                                                 squared_norm(np.expand_dims(image[r, c, :], 1), image_stacked) / (
                                                     2 * (coeff['theta_beta'] ** 2))) + \
                            coeff['w2'] * np.exp(-points_square_norm / (2 * (coeff['theta_gamma'] ** 2)))

            binary_map[r, c, :, :] = np.reshape(binary_energy, (row, col))

    return binary_map


if __name__ == '__main__':
    data, unary_energies = read_unary('12_33_s_unary.txt')
    image = Image.open('in2329-supplementary_material_10/12_33_s.bmp').convert('RGB')

    nan_count = np.sum(np.isnan(unary_energies))
    print nan_count

    isinf = np.isinf(unary_energies)
    energies_isinf = np.where(isinf, 0, unary_energies)
    print 'Max unary;', np.max(energies_isinf)

    image.load()
    image = np.asarray(image, dtype=np.float64) / 255
    row, col, _ = image.shape

    # binary_map = precompute_binary_map(image, '12_33_s')

    binary_energy = np.memmap('in2329-supplementary_material_10/12_33_s.memmap', mode='r', shape=(row, col, row, col))

    q_old = data
    for i in xrange(5):
        print 'Iteration: ', i + 1
        q = update_q(q_old, unary_energies, binary_energy)
        print np.mean(np.abs(q - q_old))
        q_old = q

    with open('in2329-supplementary_material_10/12_33_s_segmentation', 'w') as filecontent:
        pickle.dump(q_old, filecontent, 2)
    #
    #     # print data.shape
    #     # print np.max(data), np.min(data), np.mean(data)
    #     # print np.max(unary_energies), np.min(unary_energies)
