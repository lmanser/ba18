#!/usr/bin/python
#-*- coding: utf-8 -*-
# Author: Linus Manser, 13-791-132
# date: 06.03.2018
# Additional Info: python3 (3.6.2) program

from Classes import *
import os
import re
import pandas as pd
import subprocess
import parselmouth
from shutil import rmtree
import random
import time
import numpy as np
import matplotlib.pyplot as plt
import argparse

def load_main_data(DB_PATH):
    """
    Load sound files (.wav) into system in order to process them later.

    The structure of the unzipped database .zip-file is used. This mustn't be
    changed, otherwise the relative paths won't find the relevant files to fully
    extract all data from the database. In order to map the age of the different
    speaker to the correct file in the system, the documentation files are read.
    The age of the individual speakers are calculated by taking the difference
    of the year of birth and the year of recording. The actual day of birth is not
    important for this task.

    :param DB_PATH:     path to the database (AgeingDatabaseReleaseII)
    :type DB_PATH:      str
    """
    recordings = []
    DOC_FILE_PATH = DB_PATH + "TCDSA_docs/doc_main.csv"
    MAIN_DATA_PATH = DB_PATH + "TCDSA_main/"
    # read doc-file to be able to map the metadata (in doc-file) to the actual
    # data (.wav-files)
    doc = pd.read_csv(DOC_FILE_PATH).to_dict()
    for key, name in doc["speaker"].items():
        SPEAKER_DATA_PATH = MAIN_DATA_PATH + name
        dob_speaker = int(doc["DOB"][key])
        gender_speaker = doc["sex"][key]
        for ROOT_PATH, dirs, files in os.walk(SPEAKER_DATA_PATH):
            for f in files:
                if ".wav" in f.lower():
                    year_of_recording = int(f[:4])
                    # neglect specific birthday. year is close enough to get age
                    age_at_recording = year_of_recording - dob_speaker
                    FILE_PATH = ROOT_PATH + "/" + f
                    recording = SpeechRecording(FILE_PATH, 
                                                name,
                                                age_at_recording,
                                                gender_speaker)
                    recordings.append(recording)
                    
    return recordings
    

def load_additional_data(DB_PATH):
    """
    Load additional sound files (.wav) into system in order to process them later.

    Since the file structure of the additional database is different than the
    main data, a different loading routine is necessary.

    :param DB_PATH:     path to the additional database (AgeingDatabaseReleaseII)
    :type DB_PATH:      str
    """
    recordings = []
    ADDITIONAL_DATA_PATH = DB_PATH + "TCDSA_additional/"
    DOC_FILE_PATH = ADDITIONAL_DATA_PATH + "TCDSA_ex/doc_main.csv"
    CURRENT_DOC_PATH = ""
    current_doc = {}

    for ROOT_PATH, dirs, files in os.walk(ADDITIONAL_DATA_PATH):
        if "data" in dirs:
            CURRENT_DOC_PATH = ROOT_PATH + "/doc.csv"
            CURRENT_DATA_PATH = ROOT_PATH + "/data/"
            current_doc = pd.read_csv(CURRENT_DOC_PATH).to_dict()
            for key, file_name in current_doc["Name"].items():
                if isinstance(file_name, str):
                    file_name = file_name.rstrip() + ".wav"
                    if os.path.isfile(CURRENT_DATA_PATH + file_name):
                        name = current_doc["Name"][key]
                        age_at_recording = current_doc["Age"][key]
                        gender_speaker = current_doc["sex"][key]
                        FILE_PATH = CURRENT_DATA_PATH + file_name
                        recording = SpeechRecording(FILE_PATH,
                                                    name,
                                                    age_at_recording,
                                                    gender_speaker)
                        recordings.append(recording)
    return recordings


def gather_training_data(filepath, TRAIN_PATH, TEST_PATH, m_classes, f_classes):
    df = pd.read_csv(filepath)
    # split data by gender
    male = df.loc[df["gender"] == 0]
    female = df.loc[df["gender"] == 1]
    
    # MALE
    m_array = male.values
    n_m = m_array.shape[0]
    # the number of test samples is 20% of the total number of samples
    n_test = int(0.2 * n_m)
    assert(n_m > n_test)
    # sample random row indices in order to filter the out of the training array
    idx = np.random.randint(n_m, size=n_test)
    male_test = m_array[idx,:]
    difidx = [i for i in range(0, n_m) if i not in idx]
    male_train = m_array[difidx,:]
    
    # FEMALE
    f_array = female.values
    n_f = f_array.shape[0]
    # the number of test samples is 20% of the total number of samples
    n_test = int(0.2 * n_f)
    assert(n_f > n_test)
    # sample random row indices in order to filter the out of the training array
    idx = np.random.randint(n_f, size=n_test)
    female_test = f_array[idx,:]
    difidx = [i for i in range(0, n_f) if i not in idx]
    female_train = f_array[difidx,:]
    
    print(male_test, male_train, female_test, female_train)
    
    exit()
    return df


def main():
    """
    main routine of base.py

    Checks for any training/testing data in the given directory. Unless there are
    accordingly named files in these directories, the base routine will load
    all data and process it as follows:
        1) store SpeechRecordings
        2) segment SpeechRecordings into Segments (of 1 seconds)
        3) save Segments in a temporary directory
           (this step is needed in order to extract the features for single samples
           in praat)
        4) extract the features of all saved Segments with featureExtraction.praat
           (This step takes a long time, since over 30k sample must be analyzed)
           This will result in a file with all extracted values in a single file
        5) With the help of the age mapping, replace each sample ID with it's
           corresponding age.
        6) shuffle and split the data in order to differentiate between training
           and testing data. This is done by taking 20 percent of all data as test
           data.
    The result of this routine are labelled (age) test and training files in the
    given directories. These files will be used to instantiate and train the
    AgeClassifier.
    """
    parser = argparse.ArgumentParser(description="Extract Features for Classifier")
    parser.add_argument("-r", "--resume",
                        action="store_true",
                        default=False,
                        help="In case the extraction was interrupted, start where you left from. In case of -r not set, the progress will be deleted !")
    args = parser.parse_args()
    # predefined age groups with their boundaries as tuples
    m_classes = {0:(25,37), 1:(48, 50), 2:(51, 64), 3:(65, 75), 4:(76, 85)}
    f_classes = {0:(25,37), 1:(48, 50), 2:(51, 64), 3:(65, 75), 4:(76, 85)}

    feature_names = ["age", "gender", "mfcc1_sdc", "mfcc2_sdc", "mfcc3_sdc", "mfcc4_sdc", "mfcc5_sdc", "mfcc6_sdc", "mfcc7_sdc", "mfcc8_sdc", "mfcc9_sdc", "mfcc10_sdc", "mfcc11_sdc", "mfcc12_sdc", "mfcc13_sdc",\
                    "mfcc1_d_sdc", "mfcc2_d_sdc", "mfcc3_d_sdc", "mfcc4_d_sdc", "mfcc5_d_sdc", "mfcc6_d_sdc", "mfcc7_d_sdc", "mfcc8_d_sdc", "mfcc9_d_sdc", "mfcc10_d_sdc", "mfcc11_d_sdc", "mfcc12_d_sdc", "mfcc13_d_sdc", \
                    "mfcc1_dd_sdc", "mfcc2_dd_sdc", "mfcc3_dd_sdc", "mfcc4_dd_sdc", "mfcc5_dd_sdc", "mfcc6_dd_sdc", "mfcc7_dd_sdc", "mfcc8_dd_sdc", "mfcc9_dd_sdc", "mfcc10_dd_sdc", "mfcc11_dd_sdc", "mfcc12_dd_sdc", "mfcc13_dd_sdc", \
                    "pitch_stdev", "pitch_min", "pitch_max", "pitch_range", "pitch_med", "jit_loc", "jit_loc_abs", "jit_rap", "jit_ppq5", "jit_ddp", "shim_loc", "shim_apq3","shim_apq5","shim_dda", "vlhr", "stilt", "skurt", "scog", "bandenergylow","bandenergyhigh","deltaUV","meanUV","varcoUV","speakingrate","speakingratio", \
                    "ff1", "ff2", "ff3", "ff4", "f1amp", "f2amp", "f3amp", "f4amp", "I12diff", "I23diff", "harmonicity", "f0"]
    print(len(feature_names))
    os.chdir("..")
    ROOT_PATH = os.getcwd() + "/"
    DB_PATH = ROOT_PATH + "AgeingDatabaseReleaseII/"
    APPDATA_PATH = ROOT_PATH + "appdata/"
    TRAIN_PATH = APPDATA_PATH + "train/*.txt"
    TEST_PATH = APPDATA_PATH + "test/*.txt"
    MALE_TRAIN_PATH = APPDATA_PATH + "train/m_train.txt"
    FEMALE_TRAIN_PATH = APPDATA_PATH + "train/f_train.txt"
    MALE_TEST_PATH = APPDATA_PATH + "test/m_test.txt"
    FEMALE_TEST_PATH = APPDATA_PATH + "test/f_test.txt"
    SEGMENT_PATH = APPDATA_PATH + "segments/"
    MAPPING_PATH = APPDATA_PATH + "mappings/"
    EXTRACTION_PATH = APPDATA_PATH + "ext/"
    FVAL_PATH = APPDATA_PATH + "fval/"
    PROGRESS_PATH = APPDATA_PATH + "progress/"
    PROGRESS_FILE_PATH = PROGRESS_PATH + "progress.txt"
    # check if there are existing training/testing files. if this is the case,
    # we don't want to reprocess all extractions made before.
    if not os.path.exists(TRAIN_PATH) and not os.path.exists(TEST_PATH):
        if input("There is neither training or testing data available\ndo you want to compute new data? (y/n)\nThis will take a long time (around 7 days)\n> ") != "n":
            start_t = time.time()
            # load data
            print(" + loading main data")
            main_data = load_main_data(DB_PATH)
            print(" + loading additional data")
            additional_data = load_additional_data(DB_PATH)
            data = main_data + additional_data
            try:
                os.mkdir(MAPPING_PATH)
            except FileExistsError:
                pass
    
            try:
                os.mkdir(SEGMENT_PATH)
            except FileExistsError:
                pass
                
            try:
                os.mkdir(FVAL_PATH)
            except FileExistsError:
                pass
                
            processed_recordings = []
            if args.resume:
                try:
                    os.mkdir(PROGRESS_PATH)
                except FileExistsError:
                    with open(PROGRESS_FILE_PATH) as progress:
                        processed_recordings = [line.strip() for line in progress]
            else:
                with open(PROGRESS_FILE_PATH, "w") as progress:
                    print(" ! resume flag was not chosen, the old progress status will be deleted")
                
            # fval_filepath = FVAL_PATH + "out_" + str(time.time())[5:10] + ".txt"
            fval_filepath = FVAL_PATH + "extracted_fvals.txt"
            # create new empty file in order to have a clean start for the newly
            # extracted feature values
            if not args.resume:
                with open(fval_filepath, "w") as fval_file:
                    fval_file.write(",".join(feature_names) + "\n")
                    print(" ! generated new feature value file, because the old one was not clean")

            for rec in data:
            # for rec in data:
                print(" + handling %s" % rec.path())
                if args.resume:
                    rec.extract_features(3, fval_filepath, SEGMENT_PATH, EXTRACTION_PATH, PROGRESS_PATH, PROGRESS_FILE_PATH, processed_recordings)
                else:
                    rec.extract_features(3, fval_filepath, SEGMENT_PATH, EXTRACTION_PATH, PROGRESS_PATH, PROGRESS_FILE_PATH)
            print(" + extracted features")
            print(" + gathering testing and training data for each gender")
            print(gather_training_data(fval_filepath, TRAIN_PATH, TEST_PATH, m_classes, f_classes))
            exit()
            # split data, 80 percent for training, 20 percent for testing
            # rounded to the nearest whole number, since there are no half lines
            # reset appdata i.e. remove ~/segments/  and ~/ext/ once the files
            # were analyzed. These files will no longer be of any use.
            # rmtree(APPDATA_PATH + "t/")
            rmtree(SEGMENT_PATH)
            rmtree(APPDATA_PATH + "ext/")
            end_t = time.time()
            print("elapsed time: %s s" % (end_t - start_t))
        else:
            print("program closed")

if __name__ == '__main__':
    main()