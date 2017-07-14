import numpy as np

w1 = 10
w2 = 3
theta_alpha = 80
theta_beta = 13
theta_gamma = 3


def map_energy(unary):
    if unary == 0:
        return float("inf")
    return -np.log(unary)


def squared_norm(p1, p2):
    return np.sum((np.array(p1) - np.array(p2)) ** 2)


def read_unary(filename):
    with open(filename, 'r') as filecontent:
        nbcol, nbrow, nbclass = [int(s) for s in filecontent.readline().split()]
        data_list = [float(s) for s in filecontent.readline().split()]

        energies = np.array(map(lambda unary: map_energy(unary), data_list)).reshape((nbcol, nbrow, nbclass))

        data = np.array(data_list).reshape((nbcol, nbrow, nbclass))
    return data, energies


def compute_binary(point1, point2, image):
    points_square_norm = squared_norm(point1, point2)
    return w1 * np.exp(-points_square_norm / (2 * (theta_alpha ** 2)) -
                       squared_norm(image[point1], image[point2]) / (2 * (theta_beta ** 2))) + \
           w2 * np.exp(-points_square_norm / (2 * (theta_gamma ** 2)))


def precompute_binary_map(image, image_name):
    filename = 'in2329-supplementary_material_10/' + image_name + '.memmap'

    row, col = image.shape
    binary_map = np.memmap(filename, dtype=np.float32, mode='w+', shape=(row, col, row, col))

    for r in xrange(row):
        for c in xrange(col):

            for current_r in xrange(row):
                for current_c in xrange(col):
                    




if __name__ == '__main__':
    data, energies = read_unary('outfile.txt')
    print data.shape
    print np.max(data), np.min(data), np.mean(data)
    print np.max(energies), np.min(energies)
