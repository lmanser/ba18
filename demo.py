#!/usr/bin/python
#-*- coding: utf-8 -*-
# Author: Linus Manser, 13-791-132
# date: 03.12.2018
# Additional Info: python3 (3.6.2) program

from Classes import AgeClassifier
from base import load_class_mapping_pd
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
from sets import *

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




def main():
    """
    main routine of run.py

    This routine runs as follows:
        1) Instantiate an AgeClassifier-object in order to take in the input values
           from recording. This step is done before asking the user to record his
           voice in order to do all the set-up work beforehand. This cuts down on
           perceived waiting time once the signal was recorded.
        2) Ask the user of his/her gender, in order to choose the correct classifier.
        3) Records the users voice
        4) classifies the speech signal into one of five age classes

    All paths used for this program are relative to the root folder, which is where
    this program is located in.
    """
    parser = argparse.ArgumentParser(description="Estimate your age by speaking to me.")
    parser.add_argument("-n", "--norecording",
                        action="store_true",
                        default=False,
                        help="Don't record speech, use the latest recording as a sample (only works when a signal was previously recorded).")
    
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
    FEMALE_MAPPING_FILE_PATH = APPDATA_PATH + "mappings/f_mapping.txt"
    MALE_MAPPING_FILE_PATH = APPDATA_PATH + "mappings/m_mapping.txt"
    recording_name = "rec.wav"
    extraction_name = "ext.txt"
    if args.verbose:
        print("+ loading mapping of age groups from '%s'" % MAPPING_FILE_PATH)
    male_age_mapping = load_class_mapping_pd(MALE_MAPPING_FILE_PATH)
    female_age_mapping = load_class_mapping_pd(FEMALE_MAPPING_FILE_PATH)
    if args.verbose:
        print("+ setting up AgeClassifiers (this may take a few seconds)")

    m = AgeClassifier(ROOT_PATH, male_age_mapping, model=MODELS_PATH+"male_m_s.joblib", features=MFCC_SET, gender="m")
    f = AgeClassifier(ROOT_PATH, male_age_mapping, model=MODELS_PATH+"female_s_r.joblib", features=MFCC_SET, gender="m")
if __name__ == '__main__':
    main()