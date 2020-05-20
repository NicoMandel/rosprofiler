#!/usr/bin/env python3

# Script to run the monte carlo simulation
# CAREFUL - run this only once to generate options for the subsequent runs in file mcprocess

from AHP import ahp_mat
import numpy as np
import os.path
import pickle

if __name__=="__main__":
    ahp_mat.initrand()
    iters = 10000
    setdict = {}
    for i in range(2,5):
        ahplist = []
        for _ in range(iters):
            arr = ahp_mat.getrandarray(i)
            ahp = ahp_mat(arr)
            if ahp.getconsistency():
                ahplist.append(ahp.df.to_numpy())
        ahparr = np.asarray(ahplist)
        # print(ahparr)
        ahpset = np.unique(ahparr,axis=0)              
        setdict[i] = ahpset
        print("Finished with iteration {} of 3".format(i-1))

    parentdir = os.path.dirname(__file__)
    dname = os.path.abspath(os.path.join(parentdir, '..', 'tmp'))
    picklefile = os.path.join(dname, 'AHP-MC.pickle')
    with open(picklefile, 'wb') as f:
        pickle.dump(setdict, f)
    print("Dumping MC Data")
