#!/usr/bin/env python3

# Script to run the monte carlo simulation

from AHP import ahp_mat
import numpy as np

if __name__=="__main__":
    ahp_mat.initrand()
    iters = 100
    setdict = {}
    for i in range(2,5):
        ahplist = []
        for _ in range(iters*(i)):
            arr = ahp_mat.getrandarray(i)
            ahp = ahp_mat(arr)
            if ahp.getconsistency():
                ahplist.append(ahp.df.to_numpy())

        ahparr = np.asarray(ahplist)
        # print(ahparr)
        ahpset = np.unique(ahparr,axis=0)              
        setdict[i] = ahpset

    print("First Test Done")
