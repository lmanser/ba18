#!/usr/bin/python
#-*- coding: utf-8 -*-
# Author: Linus Manser, 13-791-132
# date: 07.12.2018
# Additional Info: python3 (3.6.2) program
#                  This file is not of any importance anymore, since most of the
#                  statistical calculations were done by predefined functions
#                  such as the confusion matrix, used for evaluation


import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import os
import re
import pandas as pd 
from collections import defaultdict
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.formula.api import ols
from matplotlib.ticker import MaxNLocator
from Classes import reverse_mapping, load_class_mapping_pd

class Datasheet(object):
    def __init__(self, filename, male_mapping, female_mapping):
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

    def get_target_and_training_data(self):
        out_df = self.data
        out_df = out_df.dropna()
        male, female = out_df.groupby("gender")
        male = male[1].drop(["gender"], axis=1)
        self.male_target = male["age_class"]
        self.male = male.drop(["age", "age_class"], axis=1)
        female = female[1].drop(["gender"], axis=1)
        self.female_target = female["age_class"]
        self.female = female.drop(["age", "age_class"], axis=1)

    def apply_chi2(self, k=10, gender="m"):
        X = self.male
        y = self.male_target
        if gender == "f":
            X = self.female
            y = self.female_target

        chi2_selector = SelectKBest(chi2, k=k)
        X_kbest_chi2 = chi2_selector.fit_transform(X, y)
        feature_list = find_corresponding_name(X_kbest_chi2, X)
        print("Feature list from chi-squared test:\n{}".format(feature_list))

    def apply_anova_fval(self, k=10, gender="m"):
        X = self.male
        y = self.male_target
        fvalue_selector = SelectKBest(f_classif, k=k)
        X_kbest_fvalue = fvalue_selector.fit_transform(X, y)
        feature_list = find_corresponding_name(X_kbest_fvalue, X)
        print("Feature list from ANVOA F-Value test:\n{}".format(feature_list))

    def __repr__(self):
        return str(self.data)

class PopPlot(object):
    def __init__(self, datasheet):
        self.ds = datasheet
        self.df = self.ds.data

    def by_age(self, min_age=20, max_age=85):
        x = []
        y = []
        for age in range(min_age, max_age):
            count = 0
            x.append(age)
            tmp = self.df.loc[self.df["age"] == float(age)]
            y.append(len(tmp))

        fig, ax = plt.subplots()
        ax.bar(x, y)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.show()

    def by_age_class(self, min_age=20, max_age=85):
        x = []
        y = []
        for ac in range(0,5):
            count = 0
            x.append(ac)
            tmp = self.df.loc[self.df["age_class"] == float(ac)]
            y.append(len(tmp))

        fig, ax = plt.subplots()
        ax.bar(x, y)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.show()

class LongitudinalFeaturePlot(object):
    def __init__(self, datasheet):
        self.datasheet = datasheet
        self.df = datasheet.data
        self.labels = []
        self.colors = ["#2C7BB6", "#D7191C"] # 0 male, 1 female

    def scatterplot(self, name, mode="mean", max_age=85, min_age=20):
        d = self.df.loc[self.df["age"] >= min_age]
        d = d.loc[d["age"] <= max_age]
        M, F = d.groupby("gender")
        male_df = M[1]
        female_df = F[1]
        male_x = male_df["age_class"].tolist()
        female_x = female_df["age_class"].tolist()
        male_y = male_df[name].tolist()
        female_y = female_df[name].tolist()
        male_labels = list(male_df)
        female_labels = list(female_df)
        male_mean_values = []
        female_mean_values = []
        male_set_x = []
        female_set_x = []
        male_m_x = []
        female_m_x = []
        for age in set(male_x):
            male_set_x.append(age)
            v = male_df.loc[male_df["age_class"] == age][name].tolist()
            if mode == "mean":
                male_m_x.append(np.mean(v))
            elif mode == "median":
                male_m_x.append(np.median(v))
        
        for age in set(female_x):
            female_set_x.append(age)
            v = female_df.loc[female_df["age_class"] == age][name].tolist()
            if mode == "mean":
                female_m_x.append(np.mean(v))
            elif mode == "median":
                female_m_x.append(np.median(v))

        fig, ax = plt.subplots()
        ax.scatter(male_x, male_y, color="blue", alpha=0.5)
        ax.scatter(female_x, female_y, color="red", alpha=0.5)
        ax.plot(male_set_x, male_m_x)
        ax.plot(female_set_x, female_m_x)
        ax.set_title(name)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.show()

    def lineplot(self, name, mode="mean", by="age"):
        f, m = self.df.groupby("gender")
        m_labels = sorted(list(set(m[1][by])))
        f_labels = sorted(list(set(f[1][by])))
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
        x_m = m_labels

        exit()
        y_m = [d["m"][x] for x in m_labels]
        x_f = f_labels
        y_f = [d["f"][x] for x in f_labels]
        ax.scatter(x_m, y_m, color=self.colors[0])
        ax.scatter(x_f, y_f, color=self.colors[1])
        z = np.polyfit(x_m, y_m, 1)
        print(z)
        exit()

        # ax.plot(x_m,p(x_m),"r--")

        # Calculate the slope and y-intercept of the trendline
        print("XM", x_m)
        print("YM", y_m)
        # clean up x_m (get rid of NaN values)
        # fit = np.polyfit(x_m,y_m,1)
        # print(fit)
        # # Add the trendline
        # yfit = [n*fit[0] for n in x_m]+fit[1]
        # print(yfit)
        # plt.scatter(x_m,y_m)
        # plt.plot(yfit,'black')
        ax.set_title(name)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.show()

    def boxplot(self, name, medianline=False):
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
        ax1.set_title("male")
        bplot1 = ax2.boxplot(f_values, labels=f_labels, meanline=True, showfliers=False, medianprops=medianprops)
        ax2.set_title("female")
        if medianline:
            ax1.plot([i + 1 for i in m_labels], [np.median(v) for v in m_values], "b--")
            ax2.plot([j + 1 for j in f_labels], [np.median(v) for v in f_values], "r--")
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


def main():
    APPDATA_PATH = os.getcwd()[:-4] + "appdata/" # assumes this script is in ~/code
    DATA_PATH = APPDATA_PATH + "fval/"
    MAPPING_PATH = APPDATA_PATH + "mappings/"

    print(" + loading age class mapping")
    m_mapping = f_mapping = load_class_mapping_pd(MAPPING_PATH + "f_mapping.txt")

    print(" + setting up datasheet")
    ds = Datasheet(DATA_PATH + "extracted_fvals.txt", m_mapping, f_mapping)
    ds.get_target_and_training_data()
    ds.apply_chi2(k=10)
    ds.apply_anova_fval(k=10)

    print(" + plotting feature data")
    plot = LongitudinalFeaturePlot(ds)
    plot.boxplot("pVO", medianline=True)
    plot.boxplot("F0_sdc_1", medianline=True)
    plot.boxplot("F0_sdc_2", medianline=True)
    plot.boxplot("F0_sdc_3", medianline=True)
    plot.boxplot("harmonicity_sdc_3", medianline=True)
    pop = PopPlot(ds)
    pop.by_age()
    pop.by_age_class()

if __name__ == '__main__':
    main()