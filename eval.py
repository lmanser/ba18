#!/usr/bin/python
#-*- coding: utf-8 -*-
# Author: Linus Manser, 13-791-132
# date: 06.03.2018
# Additional Info: python3 (3.6.2) program

from Classes import AgeClassifier
from sets import *
import os
from run import load_class_mapping_pd


def main():
    """
    blubb
    """
    os.chdir("..")
    ROOT_PATH = os.getcwd() + "/"
    APPDATA_PATH = "appdata/"
    FEMALE_MAPPING_FILE_PATH = APPDATA_PATH + "mappings/f_mapping.txt"
    MALE_MAPPING_FILE_PATH = APPDATA_PATH + "mappings/m_mapping.txt"
    MODELS_PATH = APPDATA_PATH + "models/"
    male_age_mapping = load_class_mapping_pd(MALE_MAPPING_FILE_PATH)
    female_age_mapping = load_class_mapping_pd(FEMALE_MAPPING_FILE_PATH)
    
    m_c_m = AgeClassifier(ROOT_PATH, male_age_mapping, model=MODELS_PATH+"male_m.joblib", features=MFCC_SET, gender="m")
    m_c_s = AgeClassifier(ROOT_PATH, male_age_mapping, model=MODELS_PATH+"male_s.joblib", features=SPECTRAL_SET, gender="m")
    m_c_r = AgeClassifier(ROOT_PATH, male_age_mapping, model=MODELS_PATH+"male_r.joblib", features=RHYTHM_SET, gender="m")
    m_c_m_s = AgeClassifier(ROOT_PATH, male_age_mapping, model=MODELS_PATH+"male_m_s.joblib", features=MFCC_SET+SPECTRAL_SET, gender="m")
    m_c_s_r = AgeClassifier(ROOT_PATH, male_age_mapping, model=MODELS_PATH+"male_s_r.joblib", features=SPECTRAL_SET+RHYTHM_SET, gender="m")
    m_c_r_m = AgeClassifier(ROOT_PATH, male_age_mapping, model=MODELS_PATH+"male_r_m.joblib", features=RHYTHM_SET+MFCC_SET, gender="m")
    m_c_r_m_s = AgeClassifier(ROOT_PATH, male_age_mapping, model=MODELS_PATH+"male_r_m_s.joblib", features=RHYTHM_SET+MFCC_SET+SPECTRAL_SET, gender="m")

    f_c_m = AgeClassifier(ROOT_PATH, female_age_mapping, model=MODELS_PATH+"female_m.joblib", features=MFCC_SET, gender="f")
    f_c_s = AgeClassifier(ROOT_PATH, female_age_mapping, model=MODELS_PATH+"female_s.joblib", features=SPECTRAL_SET, gender="f")
    f_c_r = AgeClassifier(ROOT_PATH, female_age_mapping, model=MODELS_PATH+"female_r.joblib", features=RHYTHM_SET, gender="f")
    f_c_m_s = AgeClassifier(ROOT_PATH, female_age_mapping, model=MODELS_PATH+"female_m_s.joblib", features=MFCC_SET+SPECTRAL_SET, gender="f")
    f_c_s_r = AgeClassifier(ROOT_PATH, female_age_mapping, model=MODELS_PATH+"female_s_r.joblib", features=SPECTRAL_SET+RHYTHM_SET, gender="f")
    f_c_r_m = AgeClassifier(ROOT_PATH, female_age_mapping, model=MODELS_PATH+"female_r_m.joblib", features=RHYTHM_SET+MFCC_SET, gender="f")
    f_c_r_m_s = AgeClassifier(ROOT_PATH, female_age_mapping, model=MODELS_PATH+"female_r_m_s.joblib", features=RHYTHM_SET+MFCC_SET+SPECTRAL_SET, gender="f")
    
    male_cs = [m_c_m, m_c_s, m_c_r, m_c_m_s, m_c_s_r, m_c_r_m, m_c_r_m_s]
    female_cs = [f_c_m, f_c_s, f_c_r, f_c_m_s, f_c_s_r, f_c_r_m, f_c_r_m_s]
    d = {0:"MFCC", 1:"SPECTRAL", 2:"RHYTHM", 3:"MFCC + SPECTRAL", 4:"SPECTRAL + RHYTHM", 5:"MFCC + RHYTHM", 6:"MFCC + SPECTRAL + RHYTM"}
    i = 0
    print(" + MALE")
    for c in male_cs:
        acc = c.print_accuracy_score()
        c.plot_confusion_matrix(title="MALE, normalized confusion matrix (overall accuracy: %f) %s" % (acc, d[i]))
        i += 1
        
    i = 0
    print(" + FEMALE")
    for c in female_cs:
        acc = c.print_accuracy_score()
        c.plot_confusion_matrix(title="FEMALE, normalized confusion matrix (overall accuracy: %f) %s" % (acc, d[i]))
        i += 1

if __name__ == '__main__':
    main()