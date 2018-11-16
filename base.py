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

def map_id_to_age(segments, age_mapping):
    """
    Return a mapping of all segment IDs to a corresponding
    age group.

    The groups were already computed in an earlier stage of this process.
    With the help of the age_mapping, this function checks to which age group
    a segment belongs and saves the corresponding age group key to the segment
    identification (number given at the segmentation step). 

    :param segments:    Segments, which will be mapped into the corresponding
                        age group.
    :type segments:     list
    :param age_mapping: mapping of age ranges to their respective age groups
    :type age_mapping:  dict
    :return:            mapping between segment IDs and their age group
    :type return:       dict
    """
    out_map = {}

    for s in segments:
        out_map[s.identification] = s.age_of_speaker
    
    return out_map

def map_id_to_age_gender_ageclass(segments, age_mapping):
    """
    Return a mapping of all segment IDs to a corresponding
    age group.

    The groups were already computed in an earlier stage of this process.
    With the help of the age_mapping, this function checks to which age group
    a segment belongs and saves the corresponding age group key to the segment
    identification (number given at the segmentation step). 

    :param segments:    Segments, which will be mapped into the corresponding
                        age group.
    :type segments:     list
    :param age_mapping: mapping of age ranges to their respective age groups
    :type age_mapping:  dict
    :return:            mapping between segment IDs and their age group
    :type return:       dict
    """
    out_map = {}
    
    for s in segments:
        
        out_map[s.identification] = (s.age_of_speaker, s.gender_of_speaker, s.age_class)
    
    return out_map

def replace_soundnames_with_age_class(APPDATA_PATH, extraction_file, mapping, out_file):
    """
    Replace the soundnames with their corresponding age. This step is done in
    order to set up the data structure for training and testing the classifier.
    Additionally, return the number of samples in the given data set. This number
    will be used to determine the sizes of train and test sets.

    The loaded data contains the soundnames as first element in each row. Since
    this name is no longer needed when setting up the AgeClassifier, it is
    replaced by the age of the corresponding speaker, which was calculated earlier.

    :param APPDATA_PATH:    path to the location of appdata folder
    :type APPDATA_PATH:     str
    :param extraction_file: name of the file with all extraction data
    :type extraction_file:  str
    :param mapping:         mapping between segment ID and the corresponding age
                            class
    :type mapping:          dict
    :param out_file:        name of the resulting file
    :type out_file:         str

    :return:                number of samples in the dataset
    :type return:           int
    """
    EXT_FILE_PATH = APPDATA_PATH + "ext/" + extraction_file
    OUT_FILE_PATH = APPDATA_PATH + "t/" + out_file
    sample_count = 0
    with open(OUT_FILE_PATH, "w") as out:
        with open(EXT_FILE_PATH, "r") as ext:
            for line in ext:
                identification = re.search(r"^(\d+), ", line)
                try:
                    identification = int(identification.group(1))
                    out_line = re.sub(r"^\d+", str(mapping[identification]), line)
                    out.write(out_line)
                    sample_count += 1
                except AttributeError:
                    # exceptions for header line, which will be handled differently
                    newline = re.sub(r"soundname, ", r"age, ", line)
                    out.write(newline)
    return sample_count
    
def replace_soundnames_with_age_and_gender(APPDATA_PATH, extraction_file, mapping, out_file):
    """
    Replace the soundnames with their corresponding age. This step is done in
    order to set up the data structure for training and testing the classifier.
    Additionally, return the number of samples in the given data set. This number
    will be used to determine the sizes of train and test sets.

    The loaded data contains the soundnames as first element in each row. Since
    this name is no longer needed when setting up the AgeClassifier, it is
    replaced by the age of the corresponding speaker, which was calculated earlier.

    :param APPDATA_PATH:    path to the location of appdata folder
    :type APPDATA_PATH:     str
    :param extraction_file: name of the file with all extraction data
    :type extraction_file:  str
    :param mapping:         mapping between segment ID and the corresponding age
                            class
    :type mapping:          dict
    :param out_file:        name of the resulting file
    :type out_file:         str

    :return:                number of samples in the dataset
    :type return:           int
    """
    EXT_FILE_PATH = APPDATA_PATH + "ext/" + extraction_file
    OUT_FILE_PATH = APPDATA_PATH + "t/" + out_file
    sample_count = 0
    with open(OUT_FILE_PATH, "w") as out:
        with open(EXT_FILE_PATH, "r") as ext:
            for line in ext:
                identification = re.search(r"^(\d+), ", line)
                try:
                    identification = int(identification.group(1))
                    try:
                        sub_txt = "%s, %s" % (str(mapping[identification][0]), str(mapping[identification][1]))
                    except KeyError:
                        sub_txt = "--undefined--, --undefined--"
                    out_line = re.sub(r"^\d+", sub_txt , line)
                    out.write(out_line)
                    sample_count += 1
                except AttributeError:
                    # exceptions for header line, which will be handled differently
                    newline = re.sub(r"soundname, ", r"age, gender, ", line)
                    out.write(newline)
    return sample_count


def split_data(APPDATA_PATH, DATA_PATH, n, TRAIN_PATH, TEST_PATH):
    """
    Split the extracted data in order to be able to differentiate between
    training data and test data.

    :param APPDATA_PATH:    path to the location of appdata folder
    :type APPDATA_PATH:     str
    :param DATA_PATH:       path to the location of the extracted data
    :type DATA_PATH:        str
    :param n:               dividing index of our list of samples
    :type n:                int
    """
    # set up paths
    try:
        os.mkdir(APPDATA_PATH + "train/")
        os.mkdir(APPDATA_PATH + "test/")
    except FileExistsError:
        pass

    lines = []
    h_line = ""
    with open(DATA_PATH, "r") as t:
        # skip header line
        header = 1
        for line in t:
            if header < 1:
                lines.append(line)
            else:
                header = 0
                h_line = line
    # split data into train and test samples by shuffling and cutting afterwards
    random.shuffle(lines)
    train_samples = lines[:n]
    test_samples = lines[n:]
    
    # write train.txt
    with open(TRAIN_PATH, "w") as train:
        for sample in train_samples:
            train.write(sample)

    # write test.txt
    with open(TEST_PATH, "w") as test:
        for sample in test_samples:
            test.write(sample)

def compute_age_distribution(data, plot=False, min_age=25, max_age=85):
    """
    Count the number of speakers for each age. This is done to
    make age classes with equal amounts of speakers per class.

    :param data:        the loaded data from the database
    :type data:         list of SpeechRecordings
    :param plot:        flag to plot the age distribution
                        (mainly needed as output for the documentation)
    :type plot:         boolean, default value is set to False

    :return:            returns a tuple with the counted ages as well
                        as the number of speakers in total
    :type return:       tuple(dict, int)
    """
    d = {}
    total = 0
    for rec in data:
        if rec.age >= min_age and rec.age <= max_age:
            total += 1
            try:
                d[rec.age] += 1
            except KeyError:
                d[rec.age] = 1
    if plot:
        plt.bar(d.keys(), d.values(), 0.5, color='black')
        plt.ylabel('number of speakers')
        plt.xlabel('age of speaker')
        plt.show()
    return d, total


def find_age_class_mapping(data_filename, min_age=25, max_age=85, classes=5):
    """
    Find a good set of age ranges in order to equalize the number of
    speakers per age class. This is done by adding the number of speakers
    per age as long until a certain threshold is met.

    :param age_distribution:    age distribution computed from the database
    :type age_distribution:     dict
    :param number_of_speakers:  total number of speakers in the database
    :type number_of_speakers:   int
    :param classes:             number of age classes, in which the speakers
                                ages are grouped
    :type classes:              int

    :return:                    age range of each age class
    :type return:               dict{(min_age, max_age), (...), ...}
    """
    ages = []
    n = 0
    with open(data_filename, "r") as input_file:
        for line in input_file:
            split = line.split(",")
            age = int(float(split[0].strip()))
            if age >= min_age and age <= max_age:
                ages.append(age)
                n += 1
    t = n//classes
    ages.sort()
    d= {}
    j = 0
    for i in range(1, classes):
        while j <= t*i:
            try:
                d[i-1].append(ages[j])
            except KeyError:
                d[i-1] = [ages[j]]
            j += 1
    # last class gets the remaining data
    d[classes-1] = ages[j+1:]
    for age_class in d.keys():
        # extract the minimum and maximum age per age class in order
        # to define the range of each age class
        d[age_class] = (min(d[age_class]), max(d[age_class]))
    return d

def split_data_by_gender(APPDATA_PATH, DATA_PATH, data_file, male_data_file, female_data_file):
    n_male = 0
    n_female = 0
    faulty_lines_count = 0
    n_lines = 0
    with open(DATA_PATH + male_data_file, "w") as male_file:
        with open(DATA_PATH + female_data_file, "w") as female_file:
            with open(DATA_PATH + data_file, "r") as d_file:
                for line in d_file:
                    if not re.search(r"--undefined--", line):
                        if re.search(r"^.*?, m,", line):
                            male_file.write(line)
                            n_male += 1
                            n_lines += 1
                        elif re.search(r"^.*?, male,", line):
                            new_line = re.sub("male", "m", line)
                            male_file.write(new_line)
                            n_male += 1
                            n_lines += 1
                        elif re.search(r"^.*?, f,", line):
                            female_file.write(line)
                            n_female += 1
                            n_lines += 1
                        elif re.search(r"^.*?, female,", line):
                            new_line = re.sub("female", "f", line)
                            female_file.write(new_line)
                            n_female += 1
                            n_lines += 1
                        else:
                            faulty_lines_count += 1
                    else:
                        faulty_lines_count += 1
        print(" ! %i / %i lines not added, because of invalid values" % (faulty_lines_count, n_lines))
        return n_male, n_female

def write_out_class_mapping(MAPPING_PATH, mapping, filename="mapping.txt"):
    """
    Write out the class mapping in order to access it in run.py.

    :param MAPPING_PATH:    path to mappings
    :type MAPPING_PATH:     str
    :param mapping:         the computed class mapping
    :type mapping:          dict
    :param filename:        name of the mapping file
                            default value is set to 'mapping.txt'
    :type filename:         str

    :return:                None
    """
    with open(MAPPING_PATH + filename, "w") as out_file:
        for i in mapping.keys():
            line = str(i) + "\t" + str(mapping[i][0]) + "\t" + str(mapping[i][1]) + "\n"
            out_file.write(line)


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
    ######
    #####
    # MISSING SOME FEATURE IN THIS LIST!
    #####
    ######
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