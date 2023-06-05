
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



def calc_stat(array):
    array = array[~np.isnan(array)]
    if len(array) >0:
        return {"max":np.max(array),"min":np.min(array),"mean":np.mean(array),"median":np.median(array),"std":np.std(array),"last":array[-1]}
    else:
        return {"max":np.nan,"min":np.nan,"mean":np.nan,"median":np.nan,"std":np.nan,"last":np.nan}

def plot_stat(stat,xdata,col,equation):
    plt.figure(dpi=300)
    plt.errorbar(xdata,stat["mean"],stat["std"],0,"-d",label="mean with std")
    plt.errorbar(xdata,stat["median"],stat["std"],0,"-d",label="median with std")
    plt.plot(xdata,stat["max"],"-o",label="max")
    plt.plot(xdata,stat["min"],"-*",label="min")
    plt.plot(xdata,stat["last"],"-+",color="black",label="last")
    plt.legend()
    plt.grid()
    plt.xlabel(col)
    plt.ylabel(r"log(MSE)")
    plt.title(f"MSE on equation {equation+1} with the change of {col}")
    plt.savefig(f"fig/{col}_m{equation+1}.jpg")


def analize(df,col,xdata_dict):
    errors = df[col][0]
    for m in range(4): # for each equation
        stat = {"max":[],"min":[],"mean":[],"median":[],"std":[],"last":[]}
        for err in errors: # for each error sampled
            new_stat = calc_stat(err[m])
            for category in stat:
                stat[category].append(new_stat[category])
        
        plot_stat(stat=stat,xdata=xdata_dict[col],col=col,equation=m)


dxdy = [1e-6,2e-6,3e-6,4e-6,5e-6,6e-6]
dz = [2e-6,5e-6,10e-6,20e-6]
maxXY = [80e-6,120e-6,160e-6,200e-6,240e-6]
maxZ = [1e-4,2e-4,3e-4,4e-4,5e-4]

xdata_dict = {"dxdy":dxdy,"dz":dz,"maxXY":maxXY,"maxZ":maxZ}


df = pd.read_pickle("errors.txt")



for col in xdata_dict:
    print(col)
    analize(df=df,col=col,xdata_dict=xdata_dict)