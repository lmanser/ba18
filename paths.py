#!/usr/bin/python
#-*- coding: utf-8 -*-
# Author: Linus Manser, 13-791-132
# date: 03.12.2018
# Additional Info: python3 (3.6.2) program
import os
        
os.chdir("..")
ROOT_PATH = os.getcwd() + "/"
DB_PATH = ROOT_PATH + "AgeingDatabaseReleaseII/"
APPDATA_PATH = ROOT_PATH + "appdata/"
TRAIN_PATH = APPDATA_PATH + "train/*.txt"
TEST_PATH = APPDATA_PATH + "test/*.txt"
MALE_TRAIN_PATH = APPDATA_PATH + "train/m_train.csv"
FEMALE_TRAIN_PATH = APPDATA_PATH + "train/f_train.csv"
MALE_TEST_PATH = APPDATA_PATH + "test/m_test.csv"
FEMALE_TEST_PATH = APPDATA_PATH + "test/f_test.csv"
SEGMENT_PATH = APPDATA_PATH + "segments/"
MAPPING_PATH = APPDATA_PATH + "mappings/"
RECORDING_PATH = APPDATA_PATH + "recordings/"
EXTRACTION_PATH = APPDATA_PATH + "extractions/"
FVAL_PATH = APPDATA_PATH + "fval/"
PROGRESS_PATH = APPDATA_PATH + "progress/"
MODELS_PATH = APPDATA_PATH + "models/"
PROGRESS_FILE_PATH = PROGRESS_PATH + "progress.txt"
FEMALE_MAPPING_FILE_PATH = APPDATA_PATH + "mappings/f_mapping.txt"
MALE_MAPPING_FILE_PATH = APPDATA_PATH + "mappings/m_mapping.txt"

def main():
    """
    This is the collection of created paths. These paths are needed for
    collecting the relevant data when setting up the system.
    """
    print(" + Hello from paths.py")
    print(" + ROOT_PATH: %s" % ROOT_PATH)

    
if __name__ == '__main__':
    main()