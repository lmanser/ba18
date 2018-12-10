#!/usr/bin/python
#-*- coding: utf-8 -*-
# Author: Linus Manser, 13-791-132
# date: 07.12.2018
# Additional Info: python3 (3.6.2) program

from Classes import AgeClassifier
from base import add_age_class_to_data, load_class_mapping_pd, reverse_mapping
from sets import *
from paths import *
import os
import pandas as pd


def readd_age_class_to_data(filepath, male_age_mapping, female_age_mapping):
    """
    This is a make-shift function in order to be able to recompute age classes
    of the given data. This function was mainly used for debugging.

    :param filepath:            path to the csv.file containing data 
                                for a pd.DataFrame
    :type filepath:             str
    :param male_age_mapping:    mapping of male age classes
    :type male_age_mapping:     dict
    :param female_age_mapping:  mapping of female age classes
    :type female_age_mapping:   dict
    """
    male_rev_age_mapping = reverse_mapping(male_age_mapping)
    female_rev_age_mapping = reverse_mapping(female_age_mapping)
    df = pd.read_csv(filepath)
    age_classes = []
    for i in df.index:
        try:
            s = df.iloc[i]
            if s["gender"] == 0:
                try:
                    age_classes.append((i, male_rev_age_mapping[s["age"]]))
                except KeyError:
                    age_classes.append((i, 1337.0))
            else:
                try:
                    age_classes.append((i, female_rev_age_mapping[s["age"]]))
                except KeyError:
                    age_classes.append((i, 1337.1))
        except IndexError:
            pass

    for i, ac in age_classes:
        df.at[i, "age_class"] = ac

    return df

def reselect_train_test(df, MALE_TRAIN_PATH, MALE_TEST_PATH, FEMALE_TRAIN_PATH, FEMALE_TEST_PATH, limit=450):
    """
    This is a make-shift function in order to be able to reselect training and
    testing data from the given data.
    This function was mainly used for debugging.
    
    :param df:                  DataFrame containing the feature values
    :type df:                   pd.DataFrame
    :param MALE_TRAIN_PATH:     path to the male train samples
    :type MALE_TRAIN_PATH:      str
    :param MALE_TEST_PATH:      path to the male test samples
    :type MALE_TEST_PATH:       str
    :param FEMALE_TRAIN_PATH:   path to the female train samples
    :type FEMALE TRAIN_PATH:    str
    :param FEMALE_TEST_PATH:    path to the female test samples
    :type FEMALE_TEST_PATH:     str
    :param limit:               limit of number of samples per age class
    :type limit:                int
    """
    # split data by gender
    male = df.loc[df["gender"] == 0]
    female = df.loc[df["gender"] == 1]
    male_groups = male.groupby("age_class")
    female_groups = female.groupby("age_class")

    # MALE
    test_rows = []
    # take 80 percent of each age class, in order to have balanced test and training sets
    for m in male_groups:
        m = m[1]
        n_m = m.shape[0]
        n_test = int(0.2 * n_m)
        if n_test > limit:
            n_test = limit
        male_test = m.sample(n=n_test)
        for row in male_test.iterrows():
            test_rows.append(row[0])

    male_test = male.loc[test_rows,:]
    male_train = male.drop(test_rows)

    # FEMALE    
    test_rows = []
    for f in female_groups:
        f = f[1]
        n_f = f.shape[0]
        n_test = int(0.2 * n_f)
        if n_test > limit:
            n_test = limit
        female_test = f.sample(n=n_test)
        for row in female_test.iterrows():
            test_rows.append(row[0])

    female_test = female.loc[test_rows,:]
    female_train = female.drop(test_rows)

    print(male_test, male_train)
    print(female_test, female_train)

    print(" + writing testing and training files")
    print(" +   -> %s" % MALE_TRAIN_PATH)
    male_train.to_csv(MALE_TRAIN_PATH, index=False)
    print(" +   -> %s" % MALE_TEST_PATH)
    male_test.to_csv(MALE_TEST_PATH, index=False)
    print(" +   -> %s" % FEMALE_TRAIN_PATH)
    female_train.to_csv(FEMALE_TRAIN_PATH, index=False)
    print(" +   -> %s" % FEMALE_TEST_PATH)
    female_test.to_csv(FEMALE_TEST_PATH, index=False)
    print(" + collected testing and training data")

def main():
    """
    Main function of eval.py
    Script to evaluate a collection of different AgeClassifier.

    Each AgeClassifier was already created and it's model saved to ~/models.
    These models represent a selection of feature sets, which then are going to
    be compared as part of the experiment.
    """
    d = {0:"MFCC", 1:"SPECTRAL", 2:"RHYTHM", 3:"MFCC + SPECTRAL", 4:"MFCC + RHYTHM", 5:"SPECTRAL + RHYTHM", 6:"MFCC + SPECTRAL + RHYTM"}
    print(" + loading class mappings")
    male_age_mapping = load_class_mapping_pd(MALE_MAPPING_FILE_PATH)
    female_age_mapping = load_class_mapping_pd(FEMALE_MAPPING_FILE_PATH)
    print(" + setting up male AgeClassifiers")
    m_c_m = AgeClassifier(ROOT_PATH, male_age_mapping, model=MODELS_PATH+"male_m.joblib", features=MFCC_SET, gender="m")
    m_c_s = AgeClassifier(ROOT_PATH, male_age_mapping, model=MODELS_PATH+"male_s.joblib", features=SPECTRAL_SET, gender="m")
    m_c_r = AgeClassifier(ROOT_PATH, male_age_mapping, model=MODELS_PATH+"male_r.joblib", features=RHYTHM_SET, gender="m")
    m_c_m_s = AgeClassifier(ROOT_PATH, male_age_mapping, model=MODELS_PATH+"male_m_s.joblib", features=MFCC_SET+SPECTRAL_SET, gender="m")
    m_c_s_r = AgeClassifier(ROOT_PATH, male_age_mapping, model=MODELS_PATH+"male_s_r.joblib", features=SPECTRAL_SET+RHYTHM_SET, gender="m")
    m_c_r_m = AgeClassifier(ROOT_PATH, male_age_mapping, model=MODELS_PATH+"male_r_m.joblib", features=RHYTHM_SET+MFCC_SET, gender="m")
    m_c_r_m_s = AgeClassifier(ROOT_PATH, male_age_mapping, model=MODELS_PATH+"male_r_m_s.joblib", features=RHYTHM_SET+MFCC_SET+SPECTRAL_SET, gender="m")
    print(" + setting up female AgeClassifiers")
    f_c_m = AgeClassifier(ROOT_PATH, female_age_mapping, model=MODELS_PATH+"female_m.joblib", features=MFCC_SET, gender="f")
    f_c_s = AgeClassifier(ROOT_PATH, female_age_mapping, model=MODELS_PATH+"female_s.joblib", features=SPECTRAL_SET, gender="f")
    f_c_r = AgeClassifier(ROOT_PATH, female_age_mapping, model=MODELS_PATH+"female_r.joblib", features=RHYTHM_SET, gender="f")
    f_c_m_s = AgeClassifier(ROOT_PATH, female_age_mapping, model=MODELS_PATH+"female_m_s.joblib", features=MFCC_SET+SPECTRAL_SET, gender="f")
    f_c_s_r = AgeClassifier(ROOT_PATH, female_age_mapping, model=MODELS_PATH+"female_s_r.joblib", features=SPECTRAL_SET+RHYTHM_SET, gender="f")
    f_c_r_m = AgeClassifier(ROOT_PATH, female_age_mapping, model=MODELS_PATH+"female_r_m.joblib", features=RHYTHM_SET+MFCC_SET, gender="f")
    f_c_r_m_s = AgeClassifier(ROOT_PATH, female_age_mapping, model=MODELS_PATH+"female_r_m_s.joblib", features=RHYTHM_SET+MFCC_SET+SPECTRAL_SET, gender="f")
    print(" printing evaluation matrix with accuracy scores")
    male_cs = [m_c_m, m_c_s, m_c_r, m_c_m_s, m_c_r_m, m_c_s_r, m_c_r_m_s]
    female_cs = [f_c_m, f_c_s, f_c_r, f_c_m_s, f_c_r_m, f_c_s_r, f_c_r_m_s]

    i = 0
    for c in male_cs:
        acc = c.print_accuracy_score()
        c.plot_classification_report()
        c.plot_confusion_matrix(title="MALE, normalized confusion matrix (overall accuracy: %f) %s" % (acc, d[i]))
        i += 1

    i = 0
    for c in female_cs:
        acc = c.print_accuracy_score()
        c.plot_classification_report()
        c.plot_confusion_matrix(title="FEMALE, normalized confusion matrix (overall accuracy: %f) %s" % (acc, d[i]))
        i += 1

if __name__ == '__main__':
    main()