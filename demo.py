#!/usr/bin/python
#-*- coding: utf-8 -*-
# Author: Linus Manser, 13-791-132
# date: 13.12.2018
# Additional Info: python3 (3.6.2) program
# DISCLAMER: This script is only for demonstration purposes. Due to some hacks
# which increase robustness of the script, the performance of the classifier is
# lowered.

from Classes import AgeClassifier, CSVhandler
from base import load_class_mapping_pd, make_header
import librosa
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
from paths import *
from shutil import rmtree
import sidekit
import numpy as np

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
        rmtree(RECORDING_PATH)
        os.mkdir(RECORDING_PATH)
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

def process_recording_2(wave_name, RECORDING_PATH, EXTRACTION_PATH, extraction_name="recording_values.csv", gender="m"):
    """
    """
    # remove previous csv and segments, since they are not of use anymore
    try:
        os.mkdir(EXTRACTION_PATH)
    except FileExistsError:
        rmtree(EXTRACTION_PATH)
        os.mkdir(EXTRACTION_PATH)
    # gather all data and set up correct table for each segment
    try:
        subprocess.check_call(['/Applications/Praat.app/Contents/MacOS/Praat', '--run', 'code/featureExtraction.praat', '-25', '2', '0.3', '0', RECORDING_PATH, EXTRACTION_PATH, wave_name[:-4]])
    except subprocess.CalledProcessError:
        # discard segment if subprocess throws errors due to invalid data
        print(" ! Segment not processed, due to invalid values -> please try again.")
        exit()
    endings = ["_formanttable.csv", "_harmtable.csv", "_pitchtable.csv", "_rest.csv", "_spectable.csv"]
    # since MFCC data is not extracted in the best performing models (S+R), it
    # is left out for this demonstration.
    feature_values = []
    for ending in endings:
        d = CSVhandler(EXTRACTION_PATH + wave_name[:-4] + ending)
        if ending != "_rest.csv":
            for f in list(d.df):
                feature_values += d.apply_sdc(f)
        else:
            for f in list(d.df):
                if f != "soundname":
                    feature_values.append(d.df[f].values[0])

    return feature_values


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
    if args.verbose:
        print("+ setting up necessary data paths relative to '%s'" % os.getcwd())
    recording_name = "rec.wav"
    extraction_name = "ext.txt"
    if args.verbose:
        print("+ loading mapping of age groups from '%s'" % MALE_MAPPING_FILE_PATH)
    male_age_mapping = load_class_mapping_pd(MALE_MAPPING_FILE_PATH)
    female_age_mapping = load_class_mapping_pd(FEMALE_MAPPING_FILE_PATH)
    if args.verbose:
        print("+ setting up AgeClassifiers (this may take a few seconds)")

    gender = input("What is your gender? (m/f)\n> ")
    c = None
    while True:
        if gender.lower() == "m":
            # MALE
            if args.verbose:
                print(" + Setting up male classifier")
            c = AgeClassifier(ROOT_PATH, male_age_mapping, model=MODELS_PATH+"male_s_r.joblib", gender="m")
            break
        elif gender.lower() == "f":
            # FEMALE
            if args.verbose:
                print(" + Setting up female classifier")
            c = AgeClassifier(ROOT_PATH, female_age_mapping, model=MODELS_PATH+"female_s_r.joblib", gender="f")
            break
        else:
            gender = input("Please enter a valid gender (m/f)\n> ")
    duration = 3
    wave_name = "1337.wav"
    if not args.norecording:
        start = input(" + please press enter to start recording your voice for %.1f seconds (q/quit to abort)\n> " % duration)
        if start == "q" or start == "quit":
            print(" ! quit command given, terminating program")
            exit()
        else:
            record_speech(wave_name, RECORDING_PATH, chunk=1024, format=pyaudio.paInt16, channels=1, rate=16000, record_seconds=duration)
            feature_values = process_recording_2(wave_name, RECORDING_PATH, EXTRACTION_PATH, extraction_name=extraction_name, gender="m")

    else:
        feature_values = process_recording_2(wave_name, RECORDING_PATH, EXTRACTION_PATH, extraction_name=extraction_name)

    feature_values = np.asarray(feature_values).reshape(1,-1)
    print(" > " + c.predict(feature_values)[1])

if __name__ == '__main__':
    main()