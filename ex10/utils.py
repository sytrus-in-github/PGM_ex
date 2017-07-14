import numpy as np

coeff = {
'w1' : 10,
'w2' : 3,
'theta_alpha' : 80,
'theta_beta' : 13,
'theta_gamma' : 3,
}

def map_energy(unary):
    if unary == 0:
        return float("inf")
    return -np.log(unary)


def squared_norm(p1, p2):
    return np.sum((np.array(p1)-np.array(p2))**2)


def update_q(q_old, unary_energy, binary_energy):
    col, row = q_old.shape    
    q_new = np.zeros_like(q_old)
    # update q_new    
    for c in xrange(col):
        for r in xrange(row):
            q_new[c,r] = np.exp(-unary_energy - binary_energy[c,r,:,:] * q_old)
    # normalize q_new to have 1 sum
    q_new /= np.sum(q_new)
    return q_new


def read_unary(filename):
    with open(filename, 'r') as filecontent:
        nbcol, nbrow, nbclass = [int(s) for s in filecontent.readline().split()]
        data_list = [float(s) for s in filecontent.readline().split()]

        energies = np.array(map(lambda unary: map_energy(unary), data_list)).reshape((nbcol, nbrow, nbclass))

        data = np.array(data_list).reshape((nbcol, nbrow, nbclass))
    return data, energies


def compute_binary(label1, label2, point1, point2, image):
    if label1 == label2:
        return 0.

    points_square_norm = squared_norm(point1, point2)
    return w1 * np.exp(-points_square_norm / (2 * (theta_alpha ** 2)) -
                       squared_norm(image[point1], image[point2]) / (2 * (theta_beta ** 2))) + \
           w2 * np.exp(-points_square_norm / (2 * (theta_gamma ** 2)))

if __name__ == '__main__':
    data, energies = read_unary('outfile.txt')
    print data.shape
    print np.max(data), np.min(data), np.mean(data)
    print np.max(energies), np.min(energies)
