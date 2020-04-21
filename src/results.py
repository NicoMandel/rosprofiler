#!/usr/bin/env python3

import os
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt


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

def plot_df(df):
    """
    Function for plotting a single df
    """
    fig, axes = plt.subplots(2, 2, sharex=True)
    df_cpu = df.filter(regex='CPU')
    df_avail = df.filter(regex='Available')
    df_used = df.filter(regex='Used')
    used_cols = df_cpu.columns.values.tolist() + df_avail.columns.values.tolist() + df_used.columns.values.tolist()
    df_leftover = df.drop(labels=used_cols, axis=1)
    # df_memory = pd.concat([df_mem, df_swap], sort=False)
    sns.lineplot(data=df_avail, legend="full", ax=axes[0, 0])       #  dashes=False
    sns.lineplot(data=df_used, legend="full", ax=axes[0, 1])
    sns.lineplot(data=df_cpu, legend="full", ax=axes[1, 0])
    sns.lineplot(data=df_leftover, legend="full", ax=axes[1, 1])
    plt.show()


if __name__=="__main__":

    sns.set()

    # 1. get the File which results we are interested in
    counter = 1
    filetype = 'hosts'
    parentDir = os.path.dirname(__file__)
    fname = os.path.abspath(os.path.join(parentDir, '..','results','case_'+str(counter)+'_'+filetype+'.xlsx'))

    # 2. Get the dataframe from the filename
    try:
        df_dict = getDataframe(fname)
    except FileNotFoundError as e:
        print("Filename : {} Invalid, you muppet. Come on, try again.".format(fname))
    
    if len(df_dict) < 2:
        first = next(iter(df_dict))
        first_df = df_dict[first]
    else:
        # What to do with the dataframes
        for key, df in df_dict.items():
            print("\n\nKey: {}, Df: ".format(key))
            print(df.head())
            first_df = df.copy(deep=True)
            break
            # print(df.iloc[0,0:3])

    print(first_df.columns)
    # df.columns = df.columns.str.replace(' ', '_')
    plot_df(first_df)
    print("Done")