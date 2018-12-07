#!/usr/bin/python
#-*- coding: utf-8 -*-
# Author: Linus Manser, 13-791-132
# date: 06.03.2018
# Additional Info: python3 (3.6.2) program

from Classes import *
import os
import pandas as pd
from shutil import rmtree
import time
import numpy as np
import argparse
from stats import reverse_mapping

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

def update_header(filepath, header):
    """
    Replace the existing file header with the newly created one, corresponding
    to the names of the extracted feature values.

    :param filepath:    path to the file, which header must be changed
    :type filepath:     str
    :param header:      header, which will replace the old one
    :type header:       str

    :return:            no explicit return
    :type return:       None
    """
    out_lines = []
    with open(filepath, "r+") as f:
        lines = f.readlines()
        lines[0] = ",".join(header) + "\n"
        out_lines = lines
    with open(filepath, "w") as fout:
        fout.writelines(out_lines)

def add_age_class_to_data(filepath, age_mapping):
    """
    Add age class column to dataframe

    :param filepath:    path to the file, which will be augmented
    :type filepath:     str
    :param age_mapping: dataframe containing the age boundaries of each age class
    :type age_mapping:  pd.DataFrame

    :return:            no explicit return, writes the updated dataframe as
                        csv to the given filepath
    :type return:       None
    """
    rev_age_mapping = reverse_mapping(age_mapping)
    df = pd.read_csv(filepath)
    age_classes = []
    for i in df.index:
        try:
            s = df.iloc[i]
            if s["gender"] == 0:
                try:
                    age_classes.append((i, rev_age_mapping[s["age"]]))
                except KeyError:
                    age_classes.append((i, 1337.0))
            else:
                try:
                    age_classes.append((i, rev_age_mapping[s["age"]]))
                except KeyError:
                    age_classes.append((i, 1337.1))
        except IndexError:
            pass

    for i, ac in age_classes:
        df.at[i, "age_class"] = ac

    df.to_csv(filepath, index=False)

def collect_testing_training_data(filepath, MALE_TRAIN_PATH, MALE_TEST_PATH, FEMALE_TRAIN_PATH, FEMALE_TEST_PATH):
    """
    Collects testing and training data from the given data.

    :param filepath:            path to the base file, of which the data will be taken
    :type filepath:             str
    :param MALE_TRAIN_PATH:     path to the file containing male training data
    :type MALE_TRAIN_PATH:      str
    :param MALE_TEST_PATH:      path to the file containing male testing data
    :type MALE_TEST_PATH:       str
    :param FEMALE_TRAIN_PATH:   path to the file containing female training data
    :type FEMALE_TRAIN_PATH:    str
    :param FEMALE_TEST_PATH:    path to the file containing female testing data
    :type FEMALE_TEST_PATH:     str
    """
    df = pd.read_csv(filepath)
    # split data by gender
    male = df.loc[df["gender"] == 0]
    female = df.loc[df["gender"] == 1]
    # MALE
    n_m = male.shape[0]
    n_test = int(0.2 * n_m)
    male_test = male.sample(n=n_test)
    test_rows = []
    for row in male_test.iterrows():
        test_rows.append(row[0])
    male_train = male.drop(test_rows)
    # FEMALE
    n_f = female.shape[0]
    n_test = int(0.2 * n_f)
    female_test = female.sample(n=n_test)
    test_rows = []
    for row in female_test.iterrows():
        test_rows.append(row[0])
    female_train = female.drop(test_rows)
    # OUTPUT
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
            header = []
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
                
            fval_filepath = FVAL_PATH + "extracted_fvals.txt"
            # create new empty file in order to have a clean start for the newly
            # extracted feature values
            if not args.resume:
                with open(fval_filepath, "w") as fval_file:
                    fval_file.write("--UNDER CONSTRUCTION-- --UNDER CONSTRUCTION-- --UNDER CONSTRUCTION--\n")
                    print(" ! generated new feature value file, because the old one was not clean")

            for rec in data:
                print(" + handling %s" % rec.path())
                if args.resume:
                    header = rec.extract_features(3, fval_filepath, SEGMENT_PATH, EXTRACTION_PATH, PROGRESS_PATH, PROGRESS_FILE_PATH, processed_recordings)
                else:
                    header = rec.extract_features(3, fval_filepath, SEGMENT_PATH, EXTRACTION_PATH, PROGRESS_PATH, PROGRESS_FILE_PATH)

            print(" + extracted features")
            print(" + updating header of %s" % fval_filepath)
            update_header(fval_filepath, header)
            print(" + gathering testing and training data for each gender")
            # split data, 80 percent for training, 20 percent for testing
            # rounded to the nearest whole number, since there are no half lines
            add_age_class_to_data(fval_filepath)
            collect_testing_training_data(fval_filepath, MALE_TRAIN_PATH, MALE_TEST_PATH, FEMALE_TRAIN_PATH, FEMALE_TEST_PATH)

            # reset appdata i.e. remove ~/segments/  and ~/ext/ once the files
            # were analyzed. These files will no longer be of any use.
            rmtree(SEGMENT_PATH)
            rmtree(APPDATA_PATH + "ext/")
            end_t = time.time()
            print("elapsed time: %s s" % (end_t - start_t))
        else:
            print("program closed")

if __name__ == '__main__':
    main()