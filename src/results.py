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

def compare_vals(bdict, filter_list=["^(CPU L).*(max)$", "^(Used M).*(max)$", "^(Avail).*(Mem).*(min)$", "^(Swap U).*(max)$"]):
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
    filedicts["sitl_1_hosts"] = list(pc_dicts.values())[0] # renaming the pc_dicts dictionary
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
    plot_node_df_dict(resorted_dict)

    print("Done")