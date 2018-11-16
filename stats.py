#!/usr/bin/python
#-*- coding: utf-8 -*-
# Author: Linus Manser, 13-791-132
# date: 20.10.2018
# Additional Info: python3 (3.6.2) program

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import os
import re
import pandas as pd 
from collections import defaultdict
from run import load_class_mapping, load_class_mapping_pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.formula.api import ols
from matplotlib.ticker import MaxNLocator

class Datasheet(object):
    def __init__(self, filename, male_mapping, female_mapping):
        self.indexlist = ["age", "gender", "mfcc1_sdc", "mfcc2_sdc", "mfcc3_sdc", "mfcc4_sdc", "mfcc5_sdc", "mfcc6_sdc", "mfcc7_sdc", "mfcc8_sdc", "mfcc9_sdc", "mfcc10_sdc", "mfcc11_sdc", "mfcc12_sdc", "mfcc13_sdc",\
                        "mfcc1_d_sdc", "mfcc2_d_sdc", "mfcc3_d_sdc", "mfcc4_d_sdc", "mfcc5_d_sdc", "mfcc6_d_sdc", "mfcc7_d_sdc", "mfcc8_d_sdc", "mfcc9_d_sdc", "mfcc10_d_sdc", "mfcc11_d_sdc", "mfcc12_d_sdc", "mfcc13_d_sdc", \
                        "mfcc1_dd_sdc", "mfcc2_dd_sdc", "mfcc3_dd_sdc", "mfcc4_dd_sdc", "mfcc5_dd_sdc", "mfcc6_dd_sdc", "mfcc7_dd_sdc", "mfcc8_dd_sdc", "mfcc9_dd_sdc", "mfcc10_dd_sdc", "mfcc11_dd_sdc", "mfcc12_dd_sdc", "mfcc13_dd_sdc", \
                        "pitch_stdev", "pitch_min", "pitch_max", "pitch_range", "pitch_med", "jit_loc", "jit_loc_abs", "jit_rap", "jit_ppq5", "jit_ddp", "shim_loc", "shim_apq3","shim_apq5","shim_dda", "vlhr", "stilt", "skurt", "scog", "bandenergylow","bandenergyhigh","deltaUV","meanUV","varcoUV","speakingrate","speakingratio", \
                        "ff1", "ff2", "ff3", "ff4", "f1amp", "f2amp", "f3amp", "f4amp", "I12diff", "I23diff", "harmonicity", "f0"]
        self.filename = filename
        self.male_mapping = male_mapping
        self.female_mapping = female_mapping
        self.data = self._read_data()
        
    def _read_data(self):
        df = pd.read_csv(self.filename)
        m_max_index = len(self.male_mapping) - 1
        m_min_age = self.male_mapping["lowerbound"][0]
        m_max_age = self.male_mapping["upperbound"][m_max_index]
        bad_indices = []
        # sel = df[(df["age"] < m_min_age) | (df["age"] > m_max_age)]
        # for i in sel.index:
        #     bad_indices.append(i)
        # df = df.drop(set(bad_indices))
        ages = df["age"].tolist()
        genders = df["gender"].tolist()
        assert(len(ages)==len(genders))
        age_classes = []
        m_map_rev = reverse_mapping(self.male_mapping)
        f_map_rev = reverse_mapping(self.female_mapping)
        for i in df.index:
            try:
                s = df.iloc[i]
                if s["gender"] == 0:
                    try:
                        age_classes.append((i, m_map_rev[s["age"]]))
                    except KeyError:
                        age_classes.append((i, 1337.0))
                else:
                    try:
                        age_classes.append((i, f_map_rev[s["age"]]))
                    except KeyError:
                        age_classes.append((i, 1337.1))
            except IndexError:
                pass        
        for i, ac in age_classes:
            df.at[i, "age_class"] = ac
        
        for name in df:
            df[name] = df[name].astype('float64')
        return df
        
    def corr(self, method="pearson", gender=None):
        if not gender:
            return self.data.corr(method)
        elif gender == "m":
            m = self.data.loc[self.data["gender"] == 0]
            return m.corr(method)
        else:
            f = self.data.loc[self.data["gender"] == 1]
            return f.corr(method)            

    def __repr__(self):
        return str(self.data)

class PopPlot(object):
    def __init__(self, datasheet):
        self.ds = datasheet
        self.df = self.ds.data
        
    def age_samples(self):
        x = []
        y = []
        for age in range(26, 85):
            count = 0
            x.append(age)
            tmp = self.df.loc[self.df["age"] == float(age)]
            y.append(len(tmp))
            
        fig, ax = plt.subplots()
        ax.bar(x, y)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.show()
            
            
        
        
class ScatterFeaturePlot(object):
    def __init__(self, datasheet):
        self.ds = datasheet
        self.df = datasheet.data
        self.pvals_male = pvals_male
        self.pvals_female = pvals_female
        self.colors = ["#2C7BB6", "#D7191C"]

    def scatterplot(self, names, t=0.05):
        names = list(names)
        xval = 0
        x_male = []
        y_male = []
        x_female = []
        y_female = []
        for name in names:
            xval += 1
            y_male.append(abs(self.pvals_male[name]))
            y_female.append(abs(self.pvals_female[name]))
            x_male.append(name)
            x_female.append(name)
        
        fig, ax = plt.subplots()
        ax.scatter(x_male, y_male, color=self.colors[0])
        ax.scatter(x_female, y_female, color=self.colors[1])
        ax.plot([-10,10], [t, t], color="red")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.xticks(rotation=90)
        plt.show()
        
        
    
class LongitudinalFeaturePlot(object):
    def __init__(self, datasheet):
        self.datasheet = datasheet
        self.df = datasheet.data
        self.labels = []
        self.colors = ["#2C7BB6", "#D7191C"] # 0 male, 1 female
        
    def lineplot(self, name, mode="mean", by="age"):
        f, m = self.df.groupby("gender")
        m_labels = list(set(m[1][by]))
        f_labels = list(set(f[1][by]))
        d = {"m":{}, "f":{}}
        for l in m_labels:
            if mode == "mean":
                v = np.mean(m[1].loc[m[1][by] == l][name].values)
            elif mode == "median":
                v = np.median(m[1].loc[m[1][by] == l][name].values)
            d["m"][l] = v
            
        for l in f_labels:
            if mode == "mean":
                w = np.mean(f[1].loc[f[1][by] == l][name].values)
            elif mode == "median":
                w = np.median(f[1].loc[f[1][by] == l][name].values)
            d["f"][l] = w
            
        fig, ax = plt.subplots()        
        ax.plot(m_labels, [d["m"][x] for x in m_labels], color=self.colors[0])
        ax.plot(f_labels, [d["f"][x] for x in f_labels], color=self.colors[1])
        ax.set_title(name)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.show()
        

    def boxplot(self, name):
        """
        test boxplot
        """
        fig, (ax1,ax2) = plt.subplots(nrows=1, ncols=2, sharey=True)
        fig.suptitle(name)
        medianprops = dict(linestyle='--', linewidth=0.5, color="black")
        f, m = self.df.groupby("gender")
        m_grouped = m[1].groupby("age_class")
        f_grouped = f[1].groupby("age_class")
        m_labels = []
        m_values = []
        f_labels = []
        f_values = []
        for label, s in m_grouped:
            m_labels.append(label)
            x = s[name].values
            m_values.append(x)

        for label, s in f_grouped:
            f_labels.append(label)
            x = s[name].values
            f_values.append(x)
        
        if f_labels == m_labels:
            self.labels = m_labels
            
        bplot1 = ax1.boxplot(m_values, labels=m_labels, meanline=True, showfliers=False, medianprops=medianprops)
        # plt.setp(bplot1["boxes"], color="#2C7BB6")
        ax1.set_title("male")
        bplot1 = ax2.boxplot(f_values, labels=f_labels, meanline=True, showfliers=False, medianprops=medianprops)
        ax2.set_title("female")
        plt.show()
    
    def grouped_boxplot(self, name):
        """
        test boxplot
        """
        fig = plt.figure()
        ax = plt.axes()
        medianprops = dict(linestyle='--', linewidth=0.5, color="black")
        data1 = [self.datasheet.get(name,"m",i) for i in self.labels]
        data2 = [self.datasheet.get(name,"f",i) for i in self.labels]
        labels = ["male_0\nxx-xx","female_0\nxx-xx", "male_1\n25-xx", "female_1\nxx-xx", "male_2\nxx-xx", "female_2\nxx-xx", "male_3\nxx-xx", "female_3\nxx-xx", "male_4\nxx-xx", "female_4\nxx-85"]
        group_0 = [data1[0], data2[0],[],[],[],[],[],[],[],[]]
        group_1 = [[],[], data1[1], data2[1],[],[],[],[],[],[]]
        group_2 = [[],[],[],[], data1[2],data2[2],[],[],[],[]]
        group_3 = [[],[],[],[],[],[],data1[3], data2[3],[],[]]
        group_4 = [[],[],[],[],[],[],[],[],data1[4], data2[4]]
        bplot0 = ax.boxplot(group_0, labels=labels, meanline=True, showfliers=False, medianprops=medianprops)
        bplot1 = ax.boxplot(group_1, labels=labels, meanline=True, showfliers=False, medianprops=medianprops)
        bplot2 = ax.boxplot(group_2, labels=labels, meanline=True, showfliers=False, medianprops=medianprops)
        bplot3 = ax.boxplot(group_3, labels=labels, meanline=True, showfliers=False, medianprops=medianprops)
        bplot4 = ax.boxplot(group_4, labels=labels, meanline=True, showfliers=False, medianprops=medianprops)
        ax.set_title(name)
        plt.xticks(rotation=45)
        plt.show()


def find_pearson(ds, features, base="age"):
    p = ds.corr(method="pearson", gender="f")
    q = ds.corr(method="pearson", gender="m")
    male = {}
    female = {}
    for f in features:
        female[f] = p.at[base, f]
        male[f] = q.at[base, f]
        
    return male, female
    
def reverse_mapping(mapping):
    d = {}
    for index in mapping.index:
        for age in range(mapping["lowerbound"].get(index), mapping["upperbound"].get(index)+1):
            d[age] = index
    return d

def prepare_datafile(filename):
    outname = filename[:-4] + "_clean.txt"
    lines = 0
    faulty_lines = 0
    with open(outname, "w") as w:
        with open(filename, "r") as r:
            for line in r:
                lines += 1
                ignore_line = False
                line_split = line.split(",")
                for i in range(0,len(line_split)):
                    line_split[i] = line_split[i].strip()
                    if line_split[i] == "male":
                        line_split[i] = "m"
                    elif line_split[i] == "female":
                        line_split[i] = "f"
                    elif line_split[i] == "--undefined--":
                        ignore_line = True
                if not ignore_line:
                    out_line = ",".join(line_split) + "\n"
                    w.write(out_line)
                else:
                    faulty_lines += 1
    print(" ! removed %i/%i lines because of undefined values" % (faulty_lines, lines))
    return outname


def main():
    APPDATA_PATH = os.getcwd()[:-4] + "appdata/" # assumes this script is in ~/code
    DATA_PATH = APPDATA_PATH + "fval/"
    MAPPING_PATH = APPDATA_PATH + "mappings/"
    print(" + loading age class mapping")
    m_mapping = f_mapping = load_class_mapping_pd(MAPPING_PATH + "f_mapping.txt")
    # print(" + preparing datafile")
    # datafile = prepare_datafile(DATA_PATH + "t.txt")
    print(" + setting up datasheet")
    ds = Datasheet(DATA_PATH + "extracted_fvals.txt", m_mapping, f_mapping)
    print(ds)
    print(" + computing pearson correlation")
    p_male, p_female = find_pearson(ds, ds.indexlist[2:])
    
    # quick output of "significant" features
    print("M")
    for f, v in p_male.items():
        if abs(v) < 0.05:
            print("%10s\t%f" % (f,v))
    print("F")
    for f, v in p_female.items():
        if abs(v) < 0.05:
            print("%10s\t%f" % (f,v))

    # long = LongitudinalFeaturePlot(ds)
    # scatter = ScatterFeaturePlot(p_male, p_female)
    # scatter.scatterplot(p_male.keys())
    # long.lineplot("pVO")
    # long.lineplot("pVO", mode="mean", by="age_class")
    # long.boxplot("pVO")
    pl = PopPlot(ds)
    pl.age_samples()


if __name__ == '__main__':
    main()