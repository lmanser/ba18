#!/usr/bin/python
#-*- coding: utf-8 -*-
# Author: Linus Manser, 13-791-132
# date: 03.12.2018
# Additional Info: python3 (3.6.2) program

from Classes import AgeClassifier
from base import load_class_mapping_pd
import os
import sys
import pandas as pd
from sets import *
from paths import *

def main():
    """
    main routine of train.py

    This routine runs as follows:
        1) Load the corresponding age map for both genders (M/F)
        2) Instantiate a classifier for each combination of feature sets
        3) Each classifier will save its model to ~/models

    All paths used for this program are relative to the root folder, which is where
    this program is located in.
    """
    print("+ starting train.py")
    print("+ temporarily changing root directory")
    print("+ loading mapping of age groups from '%s'" % MAPPING_PATH)
    male_age_mapping = load_class_mapping_pd(MALE_MAPPING_FILE_PATH)
    female_age_mapping = load_class_mapping_pd(FEMALE_MAPPING_FILE_PATH)
    print("+ setting up AgeClassifiers (this may take a few seconds)")
    # MALEs
    m_c_m = AgeClassifier(ROOT_PATH, male_age_mapping, modelname="male_m.joblib", features=MFCC_SET, gender="m")
    m_c_s = AgeClassifier(ROOT_PATH, male_age_mapping, modelname="male_s.joblib", features=SPECTRAL_SET, gender="m")
    m_c_r = AgeClassifier(ROOT_PATH, male_age_mapping, modelname="male_r.joblib", features=RHYTHM_SET, gender="m")
    m_c_m_s = AgeClassifier(ROOT_PATH, male_age_mapping, modelname="male_m_s.joblib", features=MFCC_SET+SPECTRAL_SET, gender="m")
    m_c_s_r = AgeClassifier(ROOT_PATH, male_age_mapping, modelname="male_s_r.joblib", features=SPECTRAL_SET+RHYTHM_SET, gender="m")
    m_c_r_m = AgeClassifier(ROOT_PATH, male_age_mapping, modelname="male_r_m.joblib", features=RHYTHM_SET+MFCC_SET, gender="m")
    m_c_r_m_s = AgeClassifier(ROOT_PATH, male_age_mapping, modelname="male_r_m_s.joblib", features=RHYTHM_SET+MFCC_SET+SPECTRAL_SET, gender="m")
    # FEMALEs
    f_c_m = AgeClassifier(ROOT_PATH, female_age_mapping, modelname="female_m.joblib", features=MFCC_SET, gender="f")
    f_c_s = AgeClassifier(ROOT_PATH, female_age_mapping, modelname="female_s.joblib", features=SPECTRAL_SET, gender="f")
    f_c_r = AgeClassifier(ROOT_PATH, female_age_mapping, modelname="female_r.joblib", features=RHYTHM_SET, gender="f")
    f_c_m_s = AgeClassifier(ROOT_PATH, female_age_mapping, modelname="female_m_s.joblib", features=MFCC_SET+SPECTRAL_SET, gender="f")
    f_c_s_r = AgeClassifier(ROOT_PATH, female_age_mapping, modelname="female_s_r.joblib", features=SPECTRAL_SET+RHYTHM_SET, gender="f")
    f_c_r_m = AgeClassifier(ROOT_PATH, female_age_mapping, modelname="female_r_m.joblib", features=RHYTHM_SET+MFCC_SET, gender="f")
    f_c_r_m_s = AgeClassifier(ROOT_PATH, female_age_mapping, modelname="female_r_m_s.joblib", features=RHYTHM_SET+MFCC_SET+SPECTRAL_SET, gender="f")

if __name__ == '__main__':
    main()