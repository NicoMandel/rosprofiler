#!/usr/bin/env python3

# File to process the results from the MC simulation, dumped into the tmp folder

import pickle
import os.path
import pandas as pd
import numpy as np 
from collections import OrderedDict
from copy import deepcopy

def getmax(df):
    """
        Function to get the maximum values out. Returns a tuple
    """

    maxidcs = df.idxmax(axis=0)
    if len(np.unique(maxidcs.values)) != len(df.columns.values):
        maxidcs = sortbymax(df)
    tup = list(zip(maxidcs.index, maxidcs["Max"]))
    return tup

def sortbymax(df, n=5):
    """
        Function to sort by the maximum values in dataframe
    """
    
    outdf = pd.DataFrame(index=["Max"],columns=df.columns.values.tolist())
    for col in df.columns.values.tolist():
        subdf = df.nlargest(n, col)
        for idx in subdf.index.values.tolist():
            maxcol = subdf.loc[idx].idxmax(axis=1)
            if maxcol == col:
                outdf.at["Max",col] = idx
                break
    return outdf.T


if __name__=="__main__":

    locdir = os.path.dirname(__file__)
    pdir = os.path.abspath(os.path.join(locdir,'..','tmp'))
    
    pdfile = os.path.join(pdir,'collectdf.xlsx')
    collectdf = pd.read_excel(pdfile, index_col=0)

    keys = tuple(["L1", "L2", "L3.Perf", "L3.Compa"])
    options = tuple(collectdf.columns.values.tolist())
    
    wgtsf = os.path.join(pdir,'weightingsdict.pickle')
    with open(wgtsf, 'rb') as f:
        wgtsdict = pickle.load(f)

    # Get out the maximum value for each of those things - TODO: most of these are double-loaded - check that we get the one where it is the biggest
    maxrows = getmax(collectdf)

    # TODO: use the tuple coming out to build a quadruplet of name - maximum - index - and AHP leading to the results
    # Turn into 3 dictionaries
    ahp_dict = OrderedDict()
    maxrow_dict = OrderedDict()
    maxval_dict = OrderedDict()
    for t in maxrows:
        key = t[0]
        row = t[1]
        ahp_dict[key] = wgtsdict[row]
        maxrow_dict[key] = deepcopy(collectdf.loc[row].T)
        maxval_dict[key] = collectdf.loc[row].max()

    separa = "====================================================================="
    # Print the AHPs which give the best results for the cases that we have
    for key, ldict in ahp_dict.items():
        print("Option: {}, Value: {}\n".format(key, maxval_dict[key]))
        print("Other Options:\n{}\n".format(maxrow_dict[key].T))
        for k, lev in ldict.items():
            print(k)
            print(lev.df)
            print("\n")
        print(separa)

    print("Test Done")