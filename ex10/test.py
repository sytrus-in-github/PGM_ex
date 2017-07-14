import numpy as np
try:
    import cPickle as pickle
except:
    import pickle

if __name__ == '__main__':

    with open('in2329-supplementary_material_10/1_9_s_segmentation', 'r') as filecontent:
       q = pickle.load(filecontent)

    print q.shape
    print np.sum(np.isnan(q))