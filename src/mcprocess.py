#!/usr/bin/env python3
from __future__ import absolute_import

import os.path
import pickle
from AHP import ahp_mat
import pandas as pd 
import numpy as np
from collections import OrderedDict
from copy import deepcopy

def assignwts(names,mc_dict):
    """
        Function to assign weights to a dataframe of given size
        Params: a dictionary with the options, a list with the options
        returns: an ahp object.
        does not require checking for consistency, because that was done during MC
    """

    options = mc_dict[len(names)]
    idx = np.random.randint(0,high=len(options))
    arr = options[idx]
    df = pd.DataFrame(arr,index=names,columns=names)
    ahp = ahp_mat(df)
    return ahp 

def evaluation(leveldict, vals_dict, indexlist):
    """
        Evaluation function. Constructs a big df for computation. 
        Parameters: a dictionary with the relative level AHPs, a dictionary with the values for all of the options
    """

    fullcollist = ["Size", "Power", "Weight", "CPU Used", "Mem Used", "CPU Free", "Mem Free", "Faults"]

    # L1
    pow_vals = vals_dict["Power"].eigdf*leveldict["L1"].eigdf.loc["Power"]
    wt_vals = vals_dict["Weight"].eigdf*leveldict["L1"].eigdf.loc["Weight"]
    sz_vals = vals_dict["Size"].eigdf*leveldict["L1"].eigdf.loc["Size"]

    # L2
    faults_vals = vals_dict["Faults"].eigdf*leveldict["L2"].eigdf.loc["Reliability"]

    # L3
    cpu_used_vals = vals_dict["CPU Used"].eigdf*leveldict["L3.Perf"].eigdf.loc["CPU"]
    mem_used_vals = vals_dict["Mem Used"].eigdf*leveldict["L3.Perf"].eigdf.loc["Memory"]
    cpu_free_vals = vals_dict["CPU Free"].eigdf*leveldict["L3.Compa"].eigdf.loc["CPU"]
    mem_free_vals = vals_dict["Mem Free"].eigdf*leveldict["L3.Compa"].eigdf.loc["Memory"]

    # Constructing the Dataframe, to store the final results
    finaldf = pd.DataFrame(index=indexlist, columns=fullcollist)
    finaldf.at[:,"Size"] = sz_vals["Relative Weight"]
    finaldf.at[:,"Power"] = pow_vals["Relative Weight"]
    finaldf.at[:,"Weight"] = wt_vals["Relative Weight"]
    finaldf.at[:,"CPU Used"] = cpu_used_vals["Relative Weight"]
    finaldf.at[:,"Mem Used"] = mem_used_vals["Relative Weight"]
    finaldf.at[:,"CPU Free"] = cpu_free_vals["Relative Weight"]
    finaldf.at[:,"Mem Free"] = mem_free_vals["Relative Weight"]
    finaldf.at[:,"Faults"] = faults_vals["Relative Weight"]

    # Get the best value out - a tuple
    finaldf["Sum"] = finaldf.sum(axis=1)
    # Sanity check:
    # print("Sum: {}".format(finaldf["Sum"].sum()))
    best = finaldf["Sum"].idxmax()
    highest = finaldf["Sum"]
    return highest.T
    



if __name__=="__main__":
    iterations = 10000

    locdir = os.path.dirname(__file__)
    pdir = os.path.abspath(os.path.join(locdir,'..','tmp'))
    mcpfname = 'AHP-MC.pickle'
    relpfname = 'RelDict.pickle'

    mcpf = os.path.join(pdir, mcpfname)
    with open(mcpf, 'rb') as f:
        mc_dict = pickle.load(f)
        print("MC data successfully loaded")
    
    relpf = os.path.join(pdir, relpfname)
    with open(relpf, 'rb') as f:
        vals_dict = pickle.load(f)
        print("AHP data successfully loaded")

    l1names = ["Weight", "Size", "Power", "Computation"]
    l2names = ["Performance", "Compatibility","Reliability"]
    l3names = ["CPU", "Memory"]

    sumnamesdict = OrderedDict()
    sumnamesdict["L1"] = l1names
    sumnamesdict["L2"] = l2names
    sumnamesdict["L3.Perf"] = l3names
    sumnamesdict["L3.Compa"] = l3names

    # For values to get where we want them to be
    indexlist = vals_dict["Power"].df.index.tolist()
    weightingsdict = OrderedDict()
    collectdf = pd.DataFrame(np.NaN, index=range(iterations), columns=indexlist)

    for i in range(iterations):
        if (i+1)%100 == 0:
            print("Finished iteration {} of {}".format(i+1, iterations))

        leveldict = OrderedDict()
        for level, names in sumnamesdict.items():
            leveldict[level] = assignwts(names, mc_dict)
        # print("Assigned Relative Weights for all Levels. Now calculating global priority")
        
        # To keep a reference for later on, how we actually assigned the relative weights
        ahpdict = deepcopy(leveldict)

        # To calculate the globalweights
        for level, ahp in leveldict.items():
            if "L2" == level:
                ahp.eigdf = ahp.eigdf*leveldict["L1"].eigdf.loc["Computation"]
            elif "L3.Perf" == level:
                ahp.eigdf = ahp.eigdf*leveldict["L2"].eigdf.loc["Performance"]
            elif "L3.Compa" == level:
                ahp.eigdf = ahp.eigdf*leveldict["L2"].eigdf.loc["Compatibility"]
        # print("Calculated global Weights")

        # For the evaluation
        sumrow = evaluation(leveldict,vals_dict, indexlist=indexlist)
        collectdf.at[i] = sumrow
        weightingsdict[i] = ahpdict

    pfile = os.path.join(pdir, 'weightingsdict.pickle')
    if not os.path.exists(pfile):
        with open(pfile, 'wb') as f:
            pickle.dump(weightingsdict, f)
        print("File {} Dumped".format(pfile))
    else:
        print("File {} already exists".format(pfile))
    xlsf = os.path.join(pdir, 'collectdf.xlsx')
    if not os.path.exists(xlsf):    
        with pd.ExcelWriter(xlsf) as writer:
            collectdf.to_excel(writer)
        print("File {} dumped".format(xlsf))
    else:
        print("File {} already exists".format(xlsf))
    # print("Dumped files to {}".format(pdir))
    # print(collectdf)
    print("Test Done")
    