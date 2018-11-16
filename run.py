#!/usr/bin/python
#-*- coding: utf-8 -*-
# Author: Linus Manser, 13-791-132
# date: 06.03.2018
# Additional Info: python3 (3.6.2) program

from Classes import *
import os
import re
import random
import pyaudio
import sys
import wave
import subprocess
import time
import argparse
import pandas as pd

def record_speech(wave_name, RECORDING_PATH, chunk=1024, format=pyaudio.paInt16, channels=1, rate=16000, record_seconds=3):
    """
    Record audio signal and save it as a .wav-file.

    In order to process a audio signal of a speaker, it must be recorded first.

    :param wave_name:       name of the resulting .wav-file
    :type wave_name:        str
    :param RECORDING_PATH:  path to the resulting .wav-file
    :type RECORDING_PATH:   str
    :param chunk:           size of recording chunks
                            default size is set to 1024
    :type chunk:            int
    :param format:          format of the resulting .wav-file
    :type format:           pyaudio-format
                            default format is set to pyaudio.paInt16
    :param channels:        number of channels of the recording
                            default number is set to 1 (=mono)
    :type channels:         int
    :param rate:            sampling rate in Hz
                            default rate is set to 16000 (same as database signals)
    :type rate:             int
    :param record_seconds:  number of seconds of recording
                            default time is set to 3 seconds, which is the same
                            duration as the training samples.
    :type record_seconds:   int
    """
    try:
        os.mkdir(RECORDING_PATH)
    except FileExistsError:
        pass
    # set up recording environment
    p = pyaudio.PyAudio()
    wavefile = wave.open(RECORDING_PATH + wave_name, "wb")
    wavefile.setnchannels(channels)
    wavefile.setsampwidth(p.get_sample_size(format))
    wavefile.setframerate(rate)

    stream = p.open(format=format,
                    channels=channels,
                    rate=rate,
                    input=True,
                    output=True,
                    frames_per_buffer=chunk)

    print(" * recording for %i seconds" % record_seconds)
    # start recording process
    for i in range(0, int(rate / chunk * record_seconds)):
        data = stream.read(chunk)
        wavefile.writeframes(data)

    print(" * done")
    stream.stop_stream()
    stream.close()
    p.terminate()

def process_recording(wave_name, extraction_name, RECORDING_PATH, EXTRACTION_PATH):
    """
    Process the recorded audio signal. This is done with the same praat-script as
    used for processing all training/test data.

    :param wave_name:       name of the audio signal
    :type wave_name:        str
    :param extraction_name: name of the file, in which the processed value of this
                            recording are stored
    :type extraction_name:  str
    :param RECORDING_PATH:  path to recorded file
    :type RECORDING_PATH:   str
    :param EXTRACTION_PATH: path to the extraction file
    :type EXTRACTION_PATH:  str

    :return:                extracted input values for the classification
    :type return:           np.ndarray
    """
    try:
        os.mkdir(EXTRACTION_PATH)
    except FileExistsError:
        pass
    # execute the featureExtraction.praat file within praat in order to
    # extract the relevant features
    PRAAT_PATH = '/Applications/Praat.app/Contents/MacOS/Praat'
    while True:
        try:
            subprocess.call([PRAAT_PATH, '--run', 'code/featureExtraction.praat', '-25', '2', '0.3', '0', RECORDING_PATH, EXTRACTION_PATH, extraction_name])
            break
        except FileNotFoundError:
            # In case the given installation path of praat is wrong, let user
            # enter the correct path. This path will only be saved temporarily.
            PRAAT_PATH = input("The assumed installation path of praat was '%s'\nif this is not the case, please enter your own path to the installed praat version you want to use:\n> " % PRAAT_PATH)

    with open(EXTRACTION_PATH + extraction_name, "r") as ext_file:
        c = 0
        for line in ext_file:
            if c == 1:
                if not re.search(r"--undefined--", line):
                    # clean up the row (discard name of recording etc.)
                    input_row = line.rstrip().split(",")[1:]
                    # read row in order to set up the correct data structure and
                    # shape as input for the classifier
                    input_row = np.loadtxt(input_row, delimiter=",")
                    input_row = input_row.reshape(1,-1)
                    return input_row
                else:
                    # in case of --undefined-- data in the recording
                    # this can happen when the recording is not done properly
                    # e.g. voice is not loud enough, etc.
                    print("something went terribly wrong during the feature extraction, please try again.\nmake sure there is some audio input.")
                    exit()
            c += 1

def add_sample_to_file(filename, prediction, input_row):
    """
    add sample value line to file

    in order to fit the format of the training material, the extracted values
    are joined to a string and written to the file, of which the filename is
    given.

    :param filename:    name of the file to be written on
    :type filename:     str
    :param prediction:  result of the classification or correction of the user
    :type prediction:   int
    :param input_row:   array of extracted input values for the given sample
    :type input_row:    np.ndarray
    """
    with open(filename, "a") as f:
        # concatenate value of prediction with the extracted input values
        new_sample = [prediction] + input_row[0].tolist()
        for i in range(0, len(new_sample)):
            if i > 0:
                new_sample[i] = float(new_sample[i])
            new_sample[i] = str(new_sample[i])
        new_sample = "\n" + ", ".join(new_sample)
        f.write(new_sample)

def add_new_train_sample(classifier, prediction, age_mapping, input_row):
    """
    add new training sample to the training material.

    After predicting the users age, the programm will evaluate it's prediction
    by asking the user if the prediction was correct. If yes, the data will be
    stored in the same file as the other training material (subject to change).
    If no, the user must tell the program how old he/she is. With this data, a
    new sample can be inserted as training material. At this moment, there is no
    restriction on how many samples there can be for any given age group.

    :param classifier:  the AgeClassifier used for the prediction
    :type classifier:   Classes.AgeClassifier
    :param prediction:  result of the prediction
    :type prediction:   int
    :param age_mapping: mapping used for training as well. This helps to find the
                        correct label for the given age value in case the
                        prediction was wrong.
    :type age_mapping:  dict
    :param input_row:   input values for the prediction i.e. to-be training
                        values
    :type input_row:    np.ndarray
    """
    if input("Was this prediction correct?\n> ") == "y":
        add_sample_to_file(classifier.fname_train[0], prediction, input_row)
    else:
        # prediction was incorrect, ask user of his/her age in order to construct
        # a new sample
        age = int(input("How old are you?\n> "))
        for i in range(0, max(age_mapping.keys())+1):
            if age >= age_mapping[i][0] and age <= age_mapping[i][1]:
                add_sample_to_file(classifier.fname_train[0], i, input_row)

def load_class_mapping_pd(MAPPING_FILE_PATH):
    """
    loading age class mapping as pandas dataframe
    """
    names=["age_class", "lowerbound", "upperbound"]
    df = pd.read_csv(MAPPING_FILE_PATH, sep="\t", names=names)
    return df
    
def load_class_mapping(MAPPING_FILE_PATH):
    """
    Load the extracted Mapping of the computed age groups.
    :param MAPPING_FILE_PATH:   path to the mapping file (within mappings)
    :type MAPPING_FILE_PATH:    str

    :return:                    python dictionary with the group index as key
                                and age boundaries as tuples as value.
    :type return:               dict
    """
    d = {}
    with open(MAPPING_FILE_PATH, "r") as map:
        for line in map:
            l = line.rstrip().split("\t")
            d[int(l[0])] = (int(l[1]), int(l[2]))
    return d

def main():
    """
    main routine of run.py

    This routine runs as follows:
        1) Instantiate an AgeClassifier-object in order to take in the input values
           from recording. This step is done before asking the user to record his
           voice in order to do all the set-up work beforehand. This cuts down on
           perceived waiting time once the signal was recorded.
        2) Ask the user if she/he wants to record a new audio signal.
        3) if yes, record the signal, else abort the program
        4) process the recorded signal (extract features)
        5) predict the age of the recorded speaker based on the extracted
           feature values.
        6) print age prediction of the speaker as a range of age values
            (e.g. between 50 and 65 years old)
        7) evaluate the given prediction by asking the user how well it did.
            (this is just "for fun" i.e. as a test. it could be used in future
            in order to gather additional training material)

    All paths used for this program are relative to the root folder, which is where
    this program is located in.
    """
    print("Welcome!")
    parser = argparse.ArgumentParser(description="Estimate your age by speaking to me.")
    parser.add_argument("-n", "--norecording",
                        action="store_true",
                        default=False,
                        help="Don't record speech, use the latest recording as a sample (only works when a signal was previously recorded).")
    parser.add_argument("-e", "--evaluate",
                        action="store_true",
                        default=False,
                        help="Print out the evaluation. This consists of a confusion matrix and an accurary score (pop-up windows).")
    parser.add_argument("-v", "--verbose",
                        action="store_true",
                        default=False,
                        help="Print out certain script output. This can be useful for troubleshooting.")
    args = parser.parse_args()
    if args.verbose:
        print("+ starting Pythia")
        print("+ temporarily changing root directory")
    os.chdir("..")
    if args.verbose:
        print("+ setting up necessary data paths relative to '%s'" % os.getcwd())
    ROOT_PATH = os.getcwd() + "/"
    APPDATA_PATH = ROOT_PATH + "appdata/"
    RECORDING_PATH = APPDATA_PATH + "recordings/"
    EXTRACTION_PATH = APPDATA_PATH + "extractions/"
    MAPPING_FILE_PATH = APPDATA_PATH + "mappings/f_mapping.txt"
    recording_name = "rec.wav"
    extraction_name = "ext.txt"
    if args.verbose:
        print("+ loading mapping of age groups from '%s'" % MAPPING_FILE_PATH)
    age_mapping = load_class_mapping(MAPPING_FILE_PATH)
    if args.verbose:
        print("+ loaded age mapping: ", age_mapping)
        print("+ setting up AgeClassifier (this may take a few seconds)")
    c = AgeClassifier(ROOT_PATH, age_mapping)

    if args.norecording:
            if args.verbose:
                print("+ --norecording flag given. Sound file will be taken from '%s'" % RECORDING_PATH)
            input_row = process_recording(recording_name, extraction_name, RECORDING_PATH, EXTRACTION_PATH)
            prediction, statement = c.predict(input_row)
            print(statement)
    else:
        while input("If you want to record a new sample, please confirm (y)\n(immediately after confirming, you can speak into the microphone to record your speech)\n> ") == "y":
            record_speech(recording_name, RECORDING_PATH)
            input_row = process_recording(recording_name, extraction_name, RECORDING_PATH, EXTRACTION_PATH)
            prediction, statement = c.predict(input_row)
            print(statement)
            add_new_train_sample(c, prediction, age_mapping, input_row)

    if args.evaluate:
        c.print_accuracy_score()
        c.plot_confusion_matrix()
        c.plot_age_group_sizes()


if __name__ == '__main__':
    main()