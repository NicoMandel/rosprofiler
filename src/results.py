#!/usr/bin/env python3

import os
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from itertools import combinations
import socket
from difflib import SequenceMatcher
from copy import deepcopy
import re
from scipy import stats

from AHP import ahp_mat
import operator

import pickle

from functools import reduce
## Helper function to get a product
def prod(iterable):
    return reduce(operator.mul, iterable, 1)
###

def getDataframe(filename):
    """
    Function to return the dataframe  
    """
    try:
        df_dict = pd.read_excel(filename, sheet_name=None, index_col=0)

        # This gets rid of the column name, which makes for an empty row. If we can live with the empty row, then we can use something else here
        for df in df_dict.values():
            df.index.name = None
        return df_dict
    except FileNotFoundError as e:
        raise e

def plot_host_df(df):
    """
    Function for plotting a single host df.
    Values in a host_df:
        "Time", "Duration", "Samples", "CPU Count", "Power",
        "CPU Load mean", "CPU Load max", "CPU Load std",
        "Used Memory mean", "Used Memory max", "Used Memory std",
        "Available Memory mean", "Available Memory min", "Available Memory std", 
        "Shared Memory mean", "Shared Memory std", "Shared Memory max",
        "Swap Available mean", "Swap Available std", "Swap Available min",
        "Swap Used mean", "Swap Used std", "Swap Used max"  
    """

    fig, axes = plt.subplots(3, 2, sharex=True)
    df_cpu = df.filter(regex='CPU L')
    df_sw_avail = df.filter(regex='Swap Av')
    df_mem_avail = df.filter(regex='Available Mem')
    df_mem_used = df.filter(regex='Used Mem')
    df_sw_used = df.filter(regex='Swap Use')
    df_mem_shared = df.filter(regex='Shared')
    used_cols = df_cpu.columns.values.tolist() + df_mem_avail.columns.values.tolist() + df_mem_used.columns.values.tolist() + df_sw_avail.columns.values.tolist() \
        + df_sw_used.columns.values.tolist() + df_mem_shared.columns.values.tolist()
    df_leftover = df.drop(labels=used_cols, axis=1)
    # df_memory = pd.concat([df_mem, df_swap], sort=False)
    ax1 = sns.lineplot(data=df_mem_avail, legend="full", ax=axes[0, 0])       #  dashes=False
    ax1.set(xlabel="Samples", ylabel="in MB")
    ax1.set_title('Available Memory')
    ax2 = sns.lineplot(data=df_mem_used, legend="full", ax=axes[0, 1])
    ax2.set(xlabel="Samples", ylabel="in MB")
    ax2.set_title('Used Memory')
    ax3 = sns.lineplot(data=df_cpu, legend="full", ax=axes[1, 0])
    ax3.set(xlabel="Samples", ylabel="in %")
    ax3.set_title('CPU Usage')
    ax4 = sns.lineplot(data=df_mem_shared, legend="full", ax=axes[1, 1])
    ax4.set(xlabel="Samples", ylabel="in MB")
    ax4.set_title('Shared Memory')
    ax5 = sns.lineplot(data=df_sw_used, legend="full", ax=axes[2, 0])
    ax5.set(xlabel="Samples", ylabel="in MB")
    ax5.set_title('Used Swap')
    ax6 = sns.lineplot(data=df_sw_avail, legend="full", ax=axes[2, 1])
    ax6.set(xlabel="Samples", ylabel="in MB")
    ax6.set_title('Available Swap')
    # sns.lineplot(data=df_leftover, legend="full", ax=axes[1, 1])
    print("Leftover values: {}".format(df_leftover.columns.values.tolist()))
    plt.show()

def summarize_node_df(df_dict):
    """
        receives a dictionary of node dataframes, which should be compared.
        Returns a dictionary of dataframes compiled by similar measures.
        Values in a node_df:
            "Time", "Duration", "Samples", "CPU Count",
            "Threads", "CPU Load mean", "CPU Load max", "CPU Load std",
            "PSS mean", "PSS std", "PSS max",
            "Swap Used mean", "Swap Used std", "Swap Used max",
            "Virtual Memory mean", "Virtual Memory std", "Virtual Memory max"
    """

    swap_df = pd.DataFrame()
    cpu_df = pd.DataFrame()
    virt_df = pd.DataFrame()
    pss_df = pd.DataFrame()
    leftover_df = pd.DataFrame()
  
    for node, df in df_dict.items():
        # Drop all std deviation values
        df.drop(list(df.filter(regex=" std")), axis=1, inplace=True)
        _sw = df.filter(regex='Swap')
        _c = df.filter(regex='CPU L')
        _vir = df.filter(regex='Virtual')
        _pss = df.filter(regex='PSS')
        used_cols = _sw.columns.values.tolist() + _c.columns.values.tolist() + _vir.columns.values.tolist() + _pss.columns.values.tolist() 
        _lefto  = df.drop(labels=used_cols, axis=1)
        dfs = [_sw, _c, _vir, _pss, _lefto]
        
        # turn the names unique with the node dictionary names
        # clean the indices
        for _df in dfs:
            new = node.split('_')[1]+'_'
            colnames = [new+name for name in _df.columns.values.tolist()]
            _df.columns = colnames
            _df.index = np.round(_df.index.to_series().to_numpy(), decimals=1)
        
        swap_df = pd.concat([swap_df, _sw], axis=1, sort=False)
        cpu_df = pd.concat([cpu_df, _c], axis=1, sort=False)
        virt_df = pd.concat([virt_df, _vir], axis=1, sort=False)
        pss_df = pd.concat([pss_df, _pss], axis=1, sort=False)
        leftover_df = pd.concat([leftover_df, _lefto], axis=1, sort=False)

    print("Test")
    compiled_df = {}
    compiled_df["Swap"] = swap_df
    compiled_df["CPU"] = cpu_df
    compiled_df["Virt"] = virt_df
    compiled_df["Pss"] = pss_df
    compiled_df["Leftovers"] = leftover_df
    return compiled_df
    
def plot_node_df_dict(df_dict, plot_name=""):
    """
     A function which takes a dictionary sorted by nodes and compiles it 
    """

    elems = int(np.ceil(np.sqrt(len(df_dict))))
    # fig = plt.figure()
    fig, axes = plt.subplots(elems, elems, sharex=True)
    axes = axes.reshape(-1)
    for i, (name, df) in enumerate(df_dict.items()):
        sns.lineplot(data=df, legend="full", dashes=False, ax=axes[i])
        axes[i].set(xlabel="Samples")
        axes[i].set_title(name)
        axes[i].set_ylim(0,)
    plt.suptitle(plot_name)
    plt.show()
    print("Test Done")


def compare_dicts(big_dict, matching="hosts"):
    """
        Function to compare a dictionary of  dictionaries for the same information in them.
        Mostly to be used with node dictionaries.
        For now only for **2** dictionary comparison
    """
    
    node_dictionaries = []
    if matching is "hosts":
        print("Matching by Hostnames")
    elif matching is "filename":
        print("Matching by filenames")
    else:
        print("No suitable matching specified. Matching By Hostnames")
        matching="hosts"

    for pair in combinations(big_dict.items(), 2):
        # pair is a key, value tuple
        node_dictionaries.append(compare_two_dicts(pair[0], pair[1], matching=matching))
    
    # Consolidating the dictionary combination
    consolid_dict = {}
    consolid_dict = deepcopy(node_dictionaries[0])
    for bdicts in node_dictionaries[1:]:
        for node, small_dicts in bdicts.items():
            # if the node is already in there
            if node in consolid_dict.keys():
                for key in small_dicts.keys():
                    if key not in consolid_dict[node].keys():
                        new_dict = consolid_dict[node]
                        new_dict[key] = small_dicts[key]
                        consolid_dict[node] = new_dict
            else:
                consolid_dict[node] = small_dicts
    return consolid_dict



def compare_two_dicts(dict_1, dict_2, matching="hosts"):
    """
     Granular function comparing the keys for two dictionaries to see if they match.
     Is called by `compare_dicts` function
    """
    # dict_1 and _2 being passed in are key, value tuples - have to be indexed
    process_dict = {}
    for hn_name in dict_1[1].keys():
        namesplit = hn_name.split('_')
        h_name1 = ip_lookup(namesplit[0].lower())
        nname1 = '_'.join(namesplit[1:]).lower()
        for key in dict_2[1].keys():
            namesplit2 = key.split('_')
            h_name2 = ip_lookup(namesplit2[0].lower())
            nname2 = '_'.join(namesplit2[1:]).lower()

            # Double-sided matching process for nodes
            if (nname1 in nname2) or (nname2 in nname1):
                # reorder matching. Sort the dfs according to NODES
                # Get the DFs
                combined_dict = {}
                df_1 = dict_1[1][hn_name]
                df_2 = dict_2[1][key]
                if matching is "hosts":
                    combined_dict[h_name1] = df_1
                    combined_dict[h_name2] = df_2
                elif matching is "filename":
                    pre1 = dict_1[0].split('_')[:-1]
                    pre1.append(h_name1)
                    k1 = '_'.join(pre1)
                    pre2 = dict_2[0].split('_')[:-1]
                    pre2.append(h_name2)
                    k2 = '_'.join(pre2)
                    combined_dict[k1] = df_1
                    combined_dict[k2] = df_2
                else:
                    combined_dict[h_name1] = df_1
                    combined_dict[h_name2] = df_2
                # turn into the big dictionary entry
                nname = get_overlap(nname1, nname2)
                print("Match found: {} and {}, Consolidated into: {}".format(nname1, nname2, nname))
                process_dict[nname] = combined_dict
                break
        # [print("Match found: {}".format(nname)) for key in dict_2[1].keys() if nname in key.lower()]


    if len(process_dict) < 1:
        process_dict = None
    return process_dict

def compare_host_dicts(host_dict):
    """
        Function to compare two host dictionaries
    """
    big_dict = {}
    for c_name, h_dict in host_dict.items():
        cnamesplit = c_name.split('_')
        cname = '_'.join(cnamesplit[:-1])
        for hname, df in h_dict.items():
            hname = ip_lookup(hname)
            new_name = '_'.join([cname, hname])
            big_dict[new_name] = df


    return big_dict


def filepaths(directory_string, file_string=None):
    """
        Searches from the base directory of the file for the directory with the directory string
        and in that directory searches for the file string.
        Returns:: [str] list of full filepaths.
    """
    file_dicts = {}
    dirpath = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..')
    for path in Path(dirpath).rglob('*'+directory_string+'*'):
        if os.path.isdir(path):
            for fname in Path(path).rglob('*'+file_string+'*'):
                try:
                    f = os.path.splitext(os.path.basename(fname))[0]
                    file_dicts[f] = getDataframe(fname)
                    print("Found Dataframes for file: {} in folder {}".format(f, os.path.basename(os.path.dirname(fname))))
                except FileNotFoundError:
                    print("Could not open file {}. Continuing".format(fname))
    
    # For error catching in outer function
    if len(file_dicts) < 1:
        file_dicts = None
    return file_dicts
    

def plot_processes(big_dict, filter_list=["Swap", "CPU L", "Virtual", "PSS", "Threads"]):
    """
        plotting the same processes (Nodes) if and when they are running on different plattforms
        Requires a list of dictionaries (lods) in the following format: 
        Every item in the list is a dictionary, which holds node names. For each of these node names, there is a separate dictionary, with the key
        being the identificator of the host and the value being the dataframe with the values
    """
    node_dict = {}
    for node, ch_dict in big_dict.items():
            
            # host_dict is itself a dictionary, where the keys are the hosts and the values are the dfs
            node_dict[node] = process_resources(ch_dict, filter_list)
            print("The above values are for node: {}".format(node))
            print("===============================")

    for node, df_dict in node_dict.items():
        plot_node_df_dict(df_dict, node)

    print("Test")
    # node_dict now holds dfs sorted by: dict[node] = dict[filters] = 
            


def process_resources(dictionary, filter_list, leftover="leftover"):
    """
        A function, which takes in a dictionary with a name and a long df,
        and returns a dictionary of dfs
        Params: Filter list, with default values: Swap, CPU L, Virtual and PSS  
    """
    df_dict = {}
    for i in range(len(filter_list)+1):
        if i == len(filter_list):
            df_dict[leftover] = pd.DataFrame()
        else:
            df_dict[filter_list[i]] = pd.DataFrame()
  
    for name, df in dictionary.items():
        # Drop all std deviation values
        df.reset_index(drop=True, inplace=True)       # has to be done, because the time is not aligned across the hosts
        df.drop(list(df.filter(regex=" std")), axis=1, inplace=True)
        # used_cols = []
        df.index = np.round(df.index.to_series().to_numpy(), decimals=1)
        for fil in filter_list:
            _tmp = df.filter(regex=fil)
            df.drop(_tmp.columns.values.tolist(), axis=1, inplace=True)
            # used_cols = used_cols + _tmp.columns.values.tolist()
            new_colnames = [name+'_'+col for col in _tmp.columns.values.tolist()]
            _tmp.columns = new_colnames
            df_dict[fil] = pd.concat([df_dict[fil], _tmp], axis=1, sort=False)
            # df.drop(_tmp.columns)
        new_colnames = [name+'_'+col for col in df.columns.values.tolist()]
        df.columns = new_colnames
        df_dict[leftover] = pd.concat([df_dict[leftover], df], axis=1, sort=False)
    
    for key, value in df_dict.items():
        print("Df for: {}".format(key))
        print(value.head())

    return df_dict

def compare_vals(bdict, filter_list=["^(CPU L).*(max)$", "^(Used).*(Mem).*(max)$", "^(Avail).*(Mem).*(min)$", "^(Swap).*(U).*(max)$"]):
    """ 
    Function to take in the big dictionary and compare values of interest.
    At Host level
    Values to compare
        * CPU Load Max
        * Phymem used Max
        * Phymem avail Min

    Hard Checks:
        * Swap Used max > 0?
    """
    # Prefilter the list 
    # Get the first df
    df = list(bdict.values())[0]
    collist = df.columns.values.tolist()
    
    fillist = filtercolumns(collist, filter_list)
    outdfdict = {}
    for fil in fillist:
        outdfdict[fil] = pd.DataFrame()
    
    # With the list of headers, filter the dfs
    for name, df in bdict.items():
        ndf = df.reset_index(drop=True)
        # ndf.index = np.round(df.index.to_series().to_numpy(), decimals=1)
        ndf = ndf.filter(items=fillist)
        
        for col in ndf.columns.values.tolist():
            fdf = ndf.filter(like=col, axis=1)
            fdf.columns = [name]
            outdfdict[col] = pd.concat([outdfdict[col], fdf], axis=1, sort=False)
            
    print("test Done")
    return outdfdict

def consolid_values(bdict, skipna=True, alpha=1e-3, perc=0.9):
    """
        A function to turn the big dictionary into a dataframe by using consolidated values
    """
    cpu_df = bdict['CPU Load max']
    normality = normtest(cpu_df.values)
    print("Normality test failed: {} of {}. Using Median".format(normality[np.where(normality<alpha)].shape[0],normality.shape[0]))
    print("Normality test Done. Proceeding with array building")
    print("Looking for max values")
    bdf = pd.DataFrame(data=None, index=list(bdict.keys()), columns=list(bdict.values())[0].columns.tolist())
    for key, df in bdict.items():
        # bdf.at[key] = df.max(skipna=skipna)
        prevals = df.values.to_numpy()
        prevals = prevals[np.nonzero(prevals)]
        vals = np.nanquantile(prevals, perc, axis=0)
        bdf.at[key] = vals

    print(bdf.head())
    print("Big Df done")
    return bdf

def getfaults(bdict, target_list):
    """
        Function to count the faults of a system.
        Calculated over the CPU load mean
        appends it to the target-df
    """
    ser = pd.Series(index=target_list, name="Faults")
    for name in target_list:
        df = bdict[name]
        ndf = df.reset_index(drop=True, inplace=False)
        ndf = ndf.filter(regex="^(CPU L).*(mean)$")
        ct = np.count_nonzero(ndf.values < 0.1)
        ser.at[name] = ct
    return ser
        

def getpower(bdict, target_list, pinom=4.1):
    """
        Function to calculate the median power usage.
        Appends it to the target df
        receives a list
        returns a Series
    """
    ser = pd.Series(index=target_list, name="Power")
    for name in target_list:
        df = bdict[name]
        ndf = df.reset_index(drop=True, inplace=False)
        ndf = ndf.filter(like="Power")
        if np.sum(ndf.values) < 0: # Since the values if the file is not found is negative, we use the CPU_Load_max
            alt = df.filter(regex="^(CPU L).*(max)")
            if "pi" in name:
                # vol = pivol
                # amp = piamp
                # watt = vol*amp
                # power = np.mean((10*watt*alt.values))
                # This gives waaay to high values, use nominal wattage instead: 4.1
                avg = np.mean(alt.values)
                power = avg * pinom * 10
            else:
                print("Error. Do not know this kind of device")
                power=-1.0
        else:
            power = np.mean(ndf.values)
        ser.at[name] = power
    return ser


def normtest(arr, alpha=5e-2, axis=0, nan_policy='omit'):
    """
        A function to test for normality - to use with the CPU Load. Returns true if the 0-Hypothesis can be rejected
        (Commonly: 0-Hypothesis: it comes from a normal distribution)
    """
    _, p = stats.normaltest(arr, axis=0, nan_policy=nan_policy)
    if p.shape[0] > 1:
        return p
    else:
        if p < alpha:
            return False
        else:
            return True
        

def filtercolumns(collist, filters=["^(CPU L).*(max)$", "^(Used M).*(max)$", "^(Avail).*(Mem).*(min)$", "^(Swap U).*(max)$"]):
    """
        Test function to find out the filters for the columns
    """

    outputl = []
    for fil in filters:
        r = re.compile(str(fil))
        outputl+=list(filter(r.search, collist))
    return outputl
    
def ip_lookup(ip):
    """
        Helper function. Looks up whether a hostname is an IP or a hostname and ALWAYS returns the hostname
    """
    if len(ip.split('.')) > 2:
        ### BAD: hardcoded lookup here:
        if ("1.120" in ip) or ("1.132" in ip):
            return "nico-nano"
        try:
            hostname = socket.gethostbyaddr(ip)
            return hostname
        except:
            return ip
    else:
        return ip

def get_overlap(s1, s2):
    """
        Helper function to get the biggest overlapping substring between the strings, from [here](https://stackoverflow.com/questions/14128763/how-to-find-the-overlap-between-2-sequences-and-return-it)
        Used to find the node name overlap
    """
    s = SequenceMatcher(None, s1, s2)
    pos_a, pos_b, size = s.find_longest_match(0, len(s1), 0, len(s2))
    return s1[pos_a:pos_a+size]

def process_faults(vec):
    """
        Function to return an array of the relative importance of faults. Uses the natural logarithm to scale accordingly
    """
    def calclns(vec):
        """
            Function to calculate the natural logarithms and return them
        """

        lnvec = np.zeros_like(vec)
        idxs = np.where(vec>=1.0)
        lnvec[idxs] = np.log(vec[idxs])
        return lnvec

    def lncomparison(lnval1, lnval2):
        """
            output array
            indices
            lnvec
        """
        # piecewise comparison
        if lnval1 == lnval2:
            return 1.0
        elif lnval1 < 0.9:
            return (9.0-lnval2)
        elif lnval2 < 0.9:
            return 1.0/(9.0-lnval1)
        else:
            out = 1.0+(max([lnval1, lnval2]) - min([lnval1, lnval2]))
            if lnval1 > lnval2:
                return 1.0/out
            else:
                return out

    lnvec = calclns(vec)
    outputarr=np.ones((lnvec.size, lnvec.size))
    maxln = np.max(lnvec)
    for i in range(lnvec.size):
        for j in range(lnvec.size):
            if i>=j:
                continue
            else:
                outputarr[i,j] = lncomparison(lnvec[i], lnvec[j])
    return outputarr

def flts_naive(vec):
    """
        Naive way of putting the vector in: if one is > 0, assign that a 9
    """
    
    outarr = np.ones((vec.size, vec.size))
    for i in range(vec.size):
        for j in range(vec.size):
            if i >=j:
                continue
            elif vec[i] == vec[j]:
                outarr[i,j] = 1.0
            elif vec[j] < 0.9:
                outarr[i,j] = 1.0/9.0            
            elif vec[i] < 0.9:
                outarr[i,j] = 9.0
            else:
                outarr[i,j] = np.log10(vec[j])/np.log10(vec[i])

    return outarr


# Comparing Power Things
def powerfromVI(curr, vol=5.0):
    """
        Function to calculate the wattage from voltage and current
    """
    return curr*vol

def powerfromNom(vol=None, amph=4.0,t0=0.33, volpcell=3.7, cells=3, safety=0.8):
    """
        calculating P from values common to UAVs. Flight time at hover, battery cells, battery voltage, nominal ampere hour capacity rating
    """
    if vol is None:
        vol = volpcell*cells
    p0 = (vol * amph * safety) / t0
    return p0
        
def flighttime(pe, p0, c):
    """
        calculating the new flight time using the additional power draw
    """
    tf = (c) / (p0 + pe)
    return tf

def powerstuff(vec, safety=0.8 , volpcell=3.7, cells=3, amph=4.0, t0=0.333):
    """
        Calculating the ballpark power draw that we are using for the uav
        params = safety capacity, nominal voltage, amperehour-rating, flight time at hover
        Calculating the relative reduce in flighttime - using hover values and reasonable assumptions
    """
    tf_vec = np.zeros_like(vec)
    pe_vec = np.zeros_like(vec)
    vec = vec / 1000.0            # to get amperes
    p0 = powerfromNom(amph=amph,t0=t0,safety=safety,cells=cells,volpcell=volpcell)
    powfunc = np.vectorize(powerfromVI)
    pe_vec = powfunc(vec)
    c = volpcell*cells*amph*safety
    tffunc = np.vectorize(flighttime)
    tf_vec = tffunc(pe_vec,p0=p0,c=c)
    diffvec = (t0-tf_vec)/t0
    arr = np.ones((vec.size, vec.size))
    for i in range(vec.size):
        for j in range(vec.size):
            if i>=j:
                continue
            else:
                arr[i,j] = 1.0/comparestraight(diffvec[i], diffvec[j])

    return arr

# Comparing Weight and Volume 
def weightcalc(w1, w2, w0=1382.0):
    """
        Just use the relative weights, with hard limits @ 9 and 1/9
    """
    w1rel = w1 / (w1+w0)
    w2rel = w2 / (w2+w0)
    relw = w1rel/w2rel
    if relw > 9.0:
        return 9.0
    elif relw < (1.0/9.0):
        return 1.0/9.0
    else:
        return relw

def weightstuff(df, dictnom):
    """
        function to calculate the relative weight
    """
    names = df.columns.values.tolist()
    outarr = np.ones((df.columns.values.shape[0], df.columns.values.shape[0]))
    w0 = dictnom["full"]
    for i in range(outarr.shape[0]):
        for j in range(outarr.shape[1]):
            if i >= j:
                continue
            else:
                if "pi" in names[i]:
                    w1 = dictnom["pi"]
                elif "nano" in names[i]:
                    w1 = dictnom["nano"]
                else:
                    raise ValueError("No known volume")
                if "pi" in names[j]:
                    w2 = dictnom["pi"]
                elif "nano" in names[j]:
                    w2 = dictnom["nano"]
                else:
                    raise ValueError("No known volume")
                outarr[i,j] = 1.0/weightcalc(w1,w2,w0)
    return outarr
    

def volstuff(df, dictnom):
    """
        using the dataframe columns and the dictionary to construct a relative array
    """
    names = df.columns.values.tolist()
    outarr = np.ones((df.columns.values.shape[0], df.columns.values.shape[0]))
    for i in range(outarr.shape[0]):
        for j in range(outarr.shape[1]):
            if i >= j:
                continue
            else:
                if "pi" in names[i]:
                    v1 = dictnom["pi"]
                elif "nano" in names[i]:
                    v1 = dictnom["nano"]
                else:
                    raise ValueError("No known volume")
                if "pi" in names[j]:
                    v2 = dictnom["pi"]
                elif "nano" in names[j]:
                    v2 = dictnom["nano"]
                else:
                    raise ValueError("No known volume")
                outarr[i,j] = volcalc(v1,v2)
    return outarr

def volcalcalt(v1,v2):
    """
        Alternative volume calculation. Using the product of all values
    """
    vol1=prod(v1)
    vol2=prod(v2)
    return comparestraight(vol1,vol2)


def volcalc(vec_1, vec_2):
    """
        Taking in 2 vectors of l x w x h values.
        Calculates the highest divide by the lowest value of each vector
    """

    v1 = volratio(vec_1)
    v2 = volratio(vec_2)
    return comparestraight(v1,v2)
    
def volratio(vec):
    """
        Function to calculate the ratio of lowest side to longest side
    """
    
    minval = np.min(vec)
    maxval = np.max(vec)
    return maxval/minval
  
def powervals(val1, val2, fact):
    """
        Function to calculate the preferred power values
    """
    
    if val1 == val2:
        return 1.0
    else:
        res = (val1/val2)
        if res<1.0:
            return fact/res
        else:
            return 1/(fact*res)

# Comparing directly
# Use for: CPU usage and CPU free
# Also for % of memory used - first divide by nominal value
# Also for free memory - use ln 
def comparestraight(val1, val2,scale=1.0):
    """
        Function for direct comparison. if values over 9, set to 9    
    """
    
    rel = scale*(val1/val2)
    if rel > 9.0:
        return 9.0
    elif rel < (1.0/9.0):
        return (1.0/9.0)
    else:
        return rel

def cpu_usage(vec):
    """
        Function to return comparable cpu usage values as an array/ uses comparestraight
    """

    outarr = np.ones((vec.size, vec.size))
    for i in range(outarr.shape[0]):
        for j in range(outarr.shape[1]):
            if i >= j:
                continue
            else:
                outarr[i,j] = comparestraight(vec[i], vec[j])
    
    return outarr


def cpu_free(vec, scale=1.0):
    """
        Function to return compared cpu free values as an array. uses comparestraight
    """

    vec = 1.0 - (vec/100)
    outarr = np.ones((vec.size, vec.size))
    lnvec = np.log(vec)
    for i in range(outarr.shape[0]):
        for j in range(outarr.shape[1]):
            if i >= j:
                continue
            else:
                outarr[i,j] = comparestraight(lnvec[i], lnvec[j], scale=scale)
    return outarr

def mem_used(df, dictnom):
    """
        Function to return compared memory used values as an array. Uses comparestraight.
        Divide by nominal value first. Params: df and dict
    """

    ser = df.loc["Used Memory max",:]
    vec = np.zeros_like(ser.values)
    cases = df.columns.values.tolist()
    for i, case in enumerate(df.columns.values.tolist()):
        val = df.loc["Used Memory max", case]
        if "pi" in case:
            nomm = dictnom["pi"]
        elif "nano" in case:
            nomm = dictnom["nano"]
        else:
            raise TypeError("Unknown Nominal memory value for this type of device")
        vec[i] = memrel(val,nomm)
    outarr = np.ones((vec.size, vec.size))
    for i in range(outarr.shape[0]):
        for j in range(outarr.shape[1]):
            if i>=j:
                continue
            else:
                outarr[i,j] = comparestraight(vec[i], vec[j])
    return outarr

def memrel(val, nom):
    """
        Function to return the % of memory used, using the nominal value        
    """
    return val/nom

def freemem(vec):
    """
        Function to return the free memory. Uses Comparestraight. Does not use the ln
    """
    lnvec = np.log(vec)
    outarr = np.ones((vec.size, vec.size))
    for i in range(outarr.shape[0]):
        for j in range(outarr.shape[1]):
            if i >= j:
                continue
            else:
                outarr[i,j] = comparestraight(lnvec[i], lnvec[j])
    
    return outarr

def freememperc(df, dictnom):
    """
        Alternative method to calculate the relative free memory using the nominal values of the devices
    """

    ser = df.loc["Available Memory min",:]
    vec = np.zeros_like(ser.values)
    cases = df.columns.values.tolist()
    for i, case in enumerate(df.columns.values.tolist()):
        val = df.loc["Available Memory min", case]
        if "pi" in case:
            nomm = dictnom["pi"]
        elif "nano" in case:
            nomm = dictnom["nano"]
        else:
            raise TypeError("Unknown Nominal memory value for this type of device")
        vec[i] = memrel(val,nomm)
    outarr = np.ones((vec.size, vec.size))
    for i in range(outarr.shape[0]):
        for j in range(outarr.shape[1]):
            if i>=j:
                continue
            else:
                outarr[i,j] = comparestraight(vec[i], vec[j])
    return outarr



if __name__=="__main__":

    sns.set()

    # 1. get the File which results we are interested in
    counter = 1
    filetype = 'node'
    result_host = '_pc'
    # result_host = '_nano'
    parentDir = os.path.dirname(__file__)
    fname = os.path.abspath(os.path.join(parentDir, '..','results'+result_host,'case_'+str(counter)+result_host+'_'+filetype+'.xlsx'))

    # 2. Get the dataframe from the filename
    # try:
    #     df_dict = getDataframe(fname)
    # except FileNotFoundError as e:
    #     print("Filename : {} Invalid, you muppet. Come on, try again.".format(fname))
    
    # if len(df_dict) < 2:
    #     first = next(iter(df_dict))
    #     first_df = df_dict[first]
    # else:
    #     # What to do with the dataframes
    #     for key, df in df_dict.items():
    #         print("\n\nKey: {}, Df: ".format(key))
    #         print(df.head())
    #         first_df = df.copy(deep=True)
    #         break
    #         # print(df.iloc[0,0:3])

    # print(first_df.columns)
    filetype = 'host'
    filedicts = filepaths("results_nano", filetype)
    pc_dicts = filepaths("results_pc", filetype)
    rasp_dicts = filepaths("results_pi", filetype)

    # Adding other dictionaries by hand
    filedicts["sitl_1_hosts"] = list(pc_dicts.values())[0] # renaming the pc_dicts dictionary
    filedicts["picompr_1_hosts"] = rasp_dicts["compr_1_hosts"]
    filedicts["piuncompr_1_hosts"] = rasp_dicts["uncompr_1_hosts"]

    # node_dicts = compare_dicts(filedicts, matching="filename")
    host_dicts = compare_host_dicts(filedicts)
    print("Something Something")
    # df.columns = df.columns.str.replace(' ', '_')
    # plot_host_df(first_df)
    # compiled_df = summarize_node_df(df_dict)
    # plot_node_df_dict(compiled_df)

    ###### New Shit below this line
    # Plotting the values from entire processes
    # plot_processes(node_dicts)

    # Compare only the values of interest in this case - for HOSTS
    resorted_dict = compare_vals(host_dicts)

    # On the resorted dict, turn it into a df and work from there
    # ser = resorted_dict['CPU Load max']["piuncompr_1_nico-pi"]
    # print(ser.head())
    # np.nanpercentile(ser.values, 0.9, axis=0)
    # print("")
    # fig, ax = plt.subplots()
    # ax.plot(ser)
    # plt.plot()
    # print("do_stuff_here")                                                        
    cdf = consolid_values(resorted_dict, perc=0.75)
    reddf = cdf.filter(like="nico", axis=1)
    # print(reddf.head())

    # get additional values out:
    # Dictionary subset:
    # subdict = {k: host_dicts.get(k) for k in reddf.columns.values.tolist()}
    # Faults - 0 drops 
    ser = getfaults(host_dicts, reddf.columns.values.tolist())
    reddf = pd.concat([reddf.transpose(), ser], axis=1).transpose()
    # print(reddf.head())
    # Power
    ser = getpower(host_dicts, reddf.columns.values.tolist())
    reddf = pd.concat([reddf.transpose(), ser], axis=1).transpose()
    print(reddf)

    # TODO: CONTINUE RIGHT HERE! put the values from the df into the functions for each column
    pivol = np.array([0.07, 0.016, 0.06])
    nanovol = np.array([0.12, 0.062, 0.06])

    dictvol = {}
    dictvol["pi"] = pivol
    dictvol["nano"] = nanovol

    dictwt = {}
    dictwt["pi"] = 0.044
    dictwt["nano"] = 0.252
    dictwt["full"] = 1.322

    dictmem = {}
    dictmem["nano"] = 4096.0
    dictmem["pi"] = 512.0
    # Getting the dataframes out - using the values from the original df
    # memfreearr = freemem(reddf.loc["Available Memory min"].to_numpy().astype(np.float64))
    memfreearr = freememperc(reddf, dictmem)
    memfreedf = pd.DataFrame(data=memfreearr, index=reddf.columns.values.tolist(), columns=reddf.columns.values.tolist())
    cpufreearr = cpu_free(reddf.loc["CPU Load max"].to_numpy().astype(np.float64))
    cpufreedf = pd.DataFrame(data=cpufreearr, index=reddf.columns.values.tolist(), columns=reddf.columns.values.tolist())
    cpuusedarr = cpu_usage(reddf.loc["CPU Load max"].to_numpy().astype(np.float64))
    cpuuseddf = pd.DataFrame(data=cpuusedarr, index=reddf.columns.values.tolist(), columns=reddf.columns.values.tolist())
    # fltsarr = process_faults(reddf.loc["Faults"].to_numpy().astype(np.float64))
    fltsarr = flts_naive(reddf.loc["Faults"].to_numpy().astype(np.float64))
    fltsdf = pd.DataFrame(data=fltsarr, columns=reddf.columns.values.tolist(), index=reddf.columns.values.tolist())
    pwrarr = powerstuff(reddf.loc["Power"].to_numpy().astype(np.float64))
    pwrdf = pd.DataFrame(data=pwrarr, columns=reddf.columns.values.tolist(), index=reddf.columns.values.tolist())
    memusedarr = mem_used(reddf, dictmem)
    memuseddf = pd.DataFrame(data=memusedarr, index=reddf.columns.values.tolist(), columns=reddf.columns.values.tolist())
    wtarr = weightstuff(reddf, dictwt)
    wtdf = pd.DataFrame(data=wtarr, index=reddf.columns.values.tolist(), columns=reddf.columns.values.tolist())
    volarr = volstuff(reddf, dictvol)
    voldf = pd.DataFrame(data=volarr, index=reddf.columns.values.tolist(), columns=reddf.columns.values.tolist())
    print("Initial test done")

    # Putting the stuff into AHPs
    # 1: Putting it into a dictionary

    df_dict = {}
    df_dict["CPU Used"] = cpuuseddf
    df_dict["Mem Used"] = memuseddf
    df_dict["CPU Free"] = cpufreedf
    df_dict["Mem Free"] = memfreedf
    df_dict["Power"] = pwrdf
    df_dict["Weight"] = wtdf
    df_dict["Size"] = voldf
    df_dict["Faults"] = fltsdf

    ahp_dict = {}
    for key, df in df_dict.items():
        ahp = ahp_mat(df.values,collist=df.columns.values.tolist(),name=key)
        # print(ahp.df)
        # print(ahp.df)
        if ahp.getconsistency():
            print("{} is consistent. Adding to dictionary".format(key))
            ahp_dict[key] = ahp
        else:
            print("{} Not consistent. Ratio: {}".format(key,ahp.consratio))
    print("Required consistency: {}".format(ahp.CONSISTENT))
    
    # Section on Pickling. RUN ONLY ONCE
    # parentdir = os.path.dirname(__file__)
    # dname = os.path.abspath(os.path.join(parentdir, '..'))
    # picklefile = os.path.join(dname, 'RelDict.pickle')
    # with open(picklefile, 'wb') as f:
    #     pickle.dump(ahp_dict, f)

    # ALL of them are consistent. Now do our own relative weighting
    l1names = ["Weight", "Size", "Power", "Computation"]
    l2names = ["Performance", "Compatibility","Reliability"]
    l3names = ["CPU", "Memory"]

    l1df = pd.DataFrame(data=np.ones((len(l1names), len(l1names))), index=l1names, columns=l1names)
    compdf = pd.DataFrame(data=np.ones((len(l2names), len(l2names))), index=l2names, columns=l2names)
    utildf = pd.DataFrame(data=np.ones((len(l3names), len(l3names))), index=l3names, columns=l3names)
    capacitydf = pd.DataFrame(data=np.ones((len(l3names), len(l3names))), index=l3names, columns=l3names)

    l1df.at["Weight", "Size"] = 2.0
    l1df.at["Weight", "Power"] = 1.0/3.0
    l1df.at["Weight", "Computation"] = 1.0/4.0
    l1df.at["Size", "Power"] = 1.0/5.0
    l1df.at["Size", "Computation"] = 1.0/7.0
    l1df.at["Power", "Computation"] = 1.0/2.0
   
    ahpl1 = ahp_mat(l1df.to_numpy(), collist=l1names)
    print(ahpl1.eigdf)
    print("Consistency of L1: {}".format(ahpl1.consratio))
    

    compdf.at["Performance", "Compatibility"] = 4.0 
    compdf.at["Performance", "Reliability" ] = 1.0/3.0 
    compdf.at["Compatibility", "Reliability" ] = 1.0/9.0

    ahpl2 = ahp_mat(compdf.to_numpy(), collist=l2names)
    print(ahpl2.eigdf)
    print("Consistency of L2: {}".format(ahpl2.consratio))

    ahpl3perf = ahp_mat(utildf.to_numpy(), collist=l3names)
    ahpl3compat = deepcopy(ahpl3perf)
    print(ahpl3perf.eigdf)
    print("Consistency of L3: {}".format(ahpl3perf.consratio))
    
    # At L3 - do twice
    # Global value for performance is performance * computation

    ahpl2.eigdf = ahpl2.eigdf*ahpl1.eigdf.loc["Computation"]
    ahpl3perf.eigdf = ahpl3perf.eigdf*ahpl2.eigdf.loc["Performance"]
    ahpl3compat.eigdf = ahpl3compat.eigdf*ahpl2.eigdf.loc["Compatibility"]
    # print(ahpl3perf.eigdf)
    # print(ahpl3compat.eigdf)


    # Calculating the global priorites is done. Now plug in the values
    # print("Relative values: {}".format(ahp_dict["CPU Used"].eigdf))
    # print("Relative Weighting: {}".format(ahpl3perf.eigdf.loc["CPU"]))
    # L3
    cpu_used_vals = ahp_dict["CPU Used"].eigdf*ahpl3perf.eigdf.loc["CPU"]
    # print("Resulting Values: {}".format(cpu_used_vals))
    mem_used_vals = ahp_dict["Mem Used"].eigdf*ahpl3perf.eigdf.loc["Memory"]
    cpu_free_vals = ahp_dict["CPU Free"].eigdf*ahpl3compat.eigdf.loc["CPU"]
    mem_free_vals = ahp_dict["Mem Free"].eigdf*ahpl3compat.eigdf.loc["Memory"]

    # L2
    faults_vals = ahp_dict["Faults"].eigdf*ahpl2.eigdf.loc["Reliability"]
    
    # L1
    pow_vals = ahp_dict["Power"].eigdf*ahpl1.eigdf.loc["Power"]
    wt_vals = ahp_dict["Weight"].eigdf*ahpl1.eigdf.loc["Weight"]
    vol_vals = ahp_dict["Size"].eigdf*ahpl1.eigdf.loc["Size"]

    defs = np.zeros((pow_vals.shape[0], 8))
    fullcollist = ["Size", "Power", "Weight", "CPU Used", "Memory Used", "CPU Free", "Memory Free", "Faults"]
    finaldf = pd.DataFrame(data=defs, index=pow_vals.index.tolist(), columns=fullcollist)
    print(finaldf)
    finaldf.at[:,"Size"] = vol_vals["Relative Weight"]

    finaldf.at[:,"Weight"] = wt_vals["Relative Weight"]
    finaldf.at[:,"Power"] = pow_vals["Relative Weight"]
    finaldf.at[:,"CPU Used"] = cpu_used_vals["Relative Weight"]
    finaldf.at[:,"Memory Used"] = mem_used_vals["Relative Weight"]
    finaldf.at[:,"CPU Free"] = cpu_free_vals["Relative Weight"]
    finaldf.at[:,"Memory Free"] = mem_free_vals["Relative Weight"]
    finaldf.at[:,"Faults"] = faults_vals["Relative Weight"]
    finaldf["Sum"] = finaldf.sum(axis=1)
    bestsol = finaldf["Sum"].idxmax()
    print("Best Option with given Parameters is: {}".format(bestsol))
    print(finaldf)
    print("Test Done")
    # fig, ax = plt.subplots()
    # linesofinterest = resorted_dict["CPU Load max"]["piuncompr_1_nico-pi"]
    # ax.plot(linesofinterest)
    # ax.set_ylim(0,)
    # plt.show()
    plot_node_df_dict(resorted_dict)

    print("Done")