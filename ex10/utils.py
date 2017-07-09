import numpy as np
def read_unary(filename):
    with open(filename, 'r') as filecontent:
        nbcol, nbrow, nbclass = [int(s) for s in filecontent.readline().split()]
        data = np.array([float(s) for s in filecontent.readline().split()]).reshape((nbcol, nbrow, nbclass))
    return data
    
if __name__ == '__main__':
    data = read_unary('outfile.txt')
    print data.shape
    print np.max(data), np.min(data), np.mean(data)*21
    print np.sum(np.max(data, axis=2)==1.0), 320*213