import numpy as np
from PIL import Image

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


def update_q(q_old, image, unary, binary):
    col, row = q_old.shape
    q_new = np.zeros_like(q_old)

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

    row, col = image.shape
    binary_map = np.memmap(filename, dtype=np.float32, mode='w+', shape=(row, col, row, col))

    image_stacked = np.reshape(image, -1)
    grid = np.mgrid[0:row, 0:col].reshape(2, -1)

    for r in xrange(row):
        print r
        for c in xrange(col):
            points_square_norm = squared_norm(grid, [[r], [c]])

            binary_energy = coeff['w1'] * np.exp(-points_square_norm / (2 * (coeff['theta_alpha'] ** 2)) -
                                                 squared_norm(image[r, c], image_stacked) / (
                                                 2 * (coeff['theta_beta'] ** 2))) + \
                            coeff['w2'] * np.exp(-points_square_norm / (2 * (coeff['theta_gamma'] ** 2)))

            binary_map[r, c, :, :] = np.reshape(binary_energy, (row, col))

    return binary_map


if __name__ == '__main__':
    data, energies = read_unary('1_9_s_unary.txt')
    image = Image.open('in2329-supplementary_material_10/1_9_s.bmp').convert('L')

    image.load()
    image = np.asarray(image, dtype=np.float32)

    binary_map = precompute_binary_map(image, '1_9_s')

    print data.shape
    print np.max(data), np.min(data), np.mean(data)
    print np.max(energies), np.min(energies)
