import numpy as np


def map_energy(unary):
    if unary == 0:
        return float("inf")
    return -np.log(unary)


def read_unary(filename):
    with open(filename, 'r') as filecontent:
        nbcol, nbrow, nbclass = [int(s) for s in filecontent.readline().split()]
        data_list = [float(s) for s in filecontent.readline().split()]

        energies = np.array(map(lambda unary: map_energy(unary), data_list)).reshape((nbcol, nbrow, nbclass))
        
        data = np.array(data_list).reshape((nbcol, nbrow, nbclass))
    return data, energies


if __name__ == '__main__':
    data, energies = read_unary('outfile.txt')
    print data.shape
    print np.max(data), np.min(data), np.mean(data)
    print np.max(energies), np.min(energies)