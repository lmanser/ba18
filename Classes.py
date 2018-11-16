#!/usr/bin/python
#-*- coding: utf-8 -*-
# Author: Linus Manser, 13-791-132
# date: 06.03.2018
# Additional Info: python3 (3.6.2) program

import parselmouth
import itertools
import subprocess
import os
import glob
import parselmouth
import numpy as np
import pandas as pd
from shutil import rmtree
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import librosa
from sdc import *

class CSVhandler(object):
    def __init__(self, csv_filename):
        self.csv_filename = csv_filename
        self.df = self.cast(pd.read_csv(self.csv_filename), float)
        
    def cast(self, df, type):
        """
        casts a panda dataframe into the given type
        """
        try:
            df = df.astype(type)
        except ValueError:
            for n in list(df):
                try:
                    l = df.loc[df[n].map(str)=="--undefined--"].index
                    df.drop(l , inplace=True)
                    
                except KeyError:
                    # no undefined found?
                    pass
            df = df.astype(type)
        return df
            
    def sdc(self, measurement):
        data = self.df[measurement].values
        data = np.reshape(data, (-1, 1))
        data = np.transpose(data)
        data = combine_data_rows(sdc(data))
        return data
    
    def tolist(self):
        # since the list is nested, I remove the nesting my accessing the first
        # and only list inside the outer list
        return self.df.values.tolist()[0]

class SpeechRecording(object):
    """
    Class to encapsulate all relevant data of a speech recording.

    :param FILE_PATH:       path to the audio file
    :type FILE_PATH:        str
    :param name:            name of the file
    :type name:             str
    :param age:             age of the speaker at recording time.
    :type age:              int
    :param gender:          gender of speaker
    :type gender:           str
    """
    i = 0
    def __init__(self, FILE_PATH, name, age, gender):
        self.file_path = FILE_PATH
        self.name = name
        self.age = age
        self.gender = gender
        self.identification = SpeechRecording.i
        SpeechRecording.i += 1
    
    def __str__(self):
        return "<SpeechRecording - name: %s | age: %i | gender: %s | ID: %i>" % (self.name, self.age, self.gender, self.identification)
    
    def path(self):
        """
        Return the path leading to this SpeechRecording instance.

        :return:            path leading to this SpeechRecording instance.
        :type return:       str
        """
        return self.file_path
    
    def segments_to_list(self):
        """
        Return the segmented SpeechRecording as a list of Segments.

        :return:            list of the segments of this SpeechRecording instance.
        :type return:       list
        """
        return self.segments
    
    def extract_features(self, interval, outfile_path ,SEGMENT_PATH, EXTRACTION_PATH, PROGRESS_PATH, PROGRESS_FILE_PATH, processed_recordings=[]):
        # remove previous csv and segments, since they are not of use anymore
        try:
            os.mkdir(EXTRACTION_PATH)
        except FileExistsError:
            rmtree(EXTRACTION_PATH)
            os.mkdir(EXTRACTION_PATH)
        try:
            os.mkdir(SEGMENT_PATH)
        except FileExistsError:
            rmtree(SEGMENT_PATH)
            os.mkdir(SEGMENT_PATH)
        # if resume-flag is not given, delete the existing "progress" status, and start with
        # a new one
                        
        segments = self.segment_wave(interval, SEGMENT_PATH)
        all_vals = []
        current_pp = ""
        # gather all data and set up correct table for each segment
        for identification, age, gender, p, pp in segments:
            if pp in processed_recordings:
                return
            else:
                current_pp = pp
            subprocess.check_call(['/Applications/Praat.app/Contents/MacOS/Praat', '--run', 'code/featureExtraction.praat', '-25', '2', '0.3', '0', SEGMENT_PATH, EXTRACTION_PATH, str(identification)])
            """
            _formanttable.csv
            _harmtable.csv
            _pitchtable.csv
            _rest.csv
            """
            rest = CSVhandler(EXTRACTION_PATH + str(identification) + "_rest.csv")
            endings = ["_formanttable.csv", "_harmtable.csv", "_pitchtable.csv"]
            # converting strings to number in order to better handle the dataframes
            if gender == "m" or gender == "male":
                gender = 0
            else:
                gender = 1
            feature_values = [age, gender]
            # calculate mfccs with librosa
            y, sr = librosa.load(p)
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            mfcc_delta = librosa.feature.delta(mfcc)
            mfcc_deltaDelta = librosa.feature.delta(mfcc, order=2)
            M = np.concatenate((mfcc,mfcc_delta,mfcc_deltaDelta))
            mfcc_data = combine_data_rows(sdc(M))
            feature_values += mfcc_data
            feature_values += rest.tolist()[1:] # removing "soundname" by doing this
            # extract and add the other features to the segmental feature data
            for ending in endings:
                d = CSVhandler(EXTRACTION_PATH + str(identification) + ending)
                for f in list(d.df):
                    feature_values.append(d.sdc(f)[0])
            all_vals.append(feature_values)
            
        
        with open(outfile_path, "a+") as out_file:
            for line in all_vals:
                line = ",".join(map(str, line)) + "\n"
                out_file.write(line)

        # add the processed recording to the progress file
        with open(PROGRESS_FILE_PATH, "a+") as prg_file:
            prg_file.write(current_pp + "\n")
        
    def segment_wave(self, interval, SEGMENT_PATH):
        """
        """
        start_time = 0
        end_time = interval
        snd = parselmouth.Sound(self.file_path) # read in audio file
        total_duration = snd.get_end_time()
        out_list = []
        while end_time <= total_duration:
            # create a new Segment with each newly cut segment.
            segment = Segment(self.name, 
                                snd.extract_part(from_time=start_time, to_time=end_time),
                                self.age,
                                self.gender,
                                self.file_path)
            segment.save_as_wav(SEGMENT_PATH)
            out_list.append( (segment.identification, segment.age_of_speaker, segment.gender_of_speaker, segment.file_path, segment.parent_path) )
            start_time = end_time
            end_time += interval
        return out_list
        
class Segment(object):
    """
    Segment class. All SpeechRecordings are segmented during preprocessing in order
    to get 3 second long segments over which the feature extraction will run.

    :param name_of_recording:   name of the segmented SpeechRecording
    :type name_of_recording:    str
    :param wave:                audio signal
    :type wave:                 parselmouth.Sound
    :param age_of_speaker:      age of the speaker at recording time
    :type age_of_speaker:       int
    """
    i = 0
    def __init__(self, name_of_recording, wave, age_of_speaker, gender_of_speaker, parent_recording_path):
        self.name_of_recording = name_of_recording
        self.wave = wave
        self.age_of_speaker = age_of_speaker
        self.gender_of_speaker = gender_of_speaker
        self.identification = Segment.i
        self.file_path = None
        self.parent_path = parent_recording_path
        Segment.i += 1

    def save_as_wav(self, SAVE_PATH):
        """
        Save this Segment as a .wav-file at the given location

        :param SAVE_PATH:   path to the location, where this segment will be saved
        :type SAVE_PATH:    str
        """
        FILE_PATH = SAVE_PATH + '%i.wav' % self.identification
        self.wave.save(FILE_PATH, parselmouth.SoundFileFormat.WAV)
        self.file_path = FILE_PATH
        
    def __str__(self):
        return "<Segment - age: %i | gender: %s | ID: %i>" % (self.age_of_speaker, self.gender_of_speaker, self.identification)    
    
    
class AgeClassifier(object):
    """
    AgeClassifier class. AgeClassifier in order to predict the age of a speaker.
    The architecture of this Multi Layer Perceptron (type of NeuralNetwork) is based on an experiment
    for wine classification (code was distributed in class).

    :param ROOT_PATH:           root path of the system
    :type ROOT_PATH:            str
    :param age_mapping:         age mapping in order to translate the output (prediction)
                                to the corresponding age group i.e. range of age values
    :param hidden_layer_sizes:  tuple of values indicating the number of neurons
                                in each hidden layer. default is (30,30,30)
    :type hidden_layer_sizes:   tuple(int, int, int)
    :param activation:          activation function used in this neural network.
                                default value is 'logistic'
    :type activation:           str
    :param max_iter:            maximum number of iterations for fitting the NN.
    :type max_iter:             int
    """
    def __init__(self, ROOT_PATH, age_mapping, hidden_layer_sizes=(30,30,30),activation='logistic', max_iter=1000):
        self.ROOT_PATH = ROOT_PATH
        self.TRAIN_PATH = self.ROOT_PATH + '/appdata/train/'
        self.TEST_PATH = self.ROOT_PATH + '/appdata/test/'
        self.fname_train = glob.glob(self.TRAIN_PATH + 'f_train_num.txt')
        self.fname_test = glob.glob(self.TEST_PATH + 'f_test_num.txt')
        self.train_input = np.loadtxt(self.fname_train[0], delimiter=",")
        self.test_input = np.loadtxt(self.fname_test[0], delimiter=",")
        self.X_train = self.train_input[:,2:51]   # Training Features
        self.Y_train = self.train_input[:,0]      # Training Targets
        self.X_test = self.test_input[:,2:51]     # Testing Features
        self.Y_test = self.test_input[:,0]        # Testing Targets
        self.scaler = StandardScaler()
        self.mlp = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes,activation=activation,max_iter=max_iter)
        self.predictions = []
        self.age_mapping = age_mapping
        # set up the actual MLP by fitting and scaling the data
        self.scaler.fit(self.X_train)
        self.X_train = self.scaler.transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)
        self.mlp.fit(self.X_train, self.Y_train)
        self.predictions = self.mlp.predict(self.X_test)

    def print_accuracy_score(self):
        """
        Return a confusion matrix of the Classifier with the given training/test data.

        :return:            string stating the accuracy of self
        """
        if self.predictions == []:
            return "you must set up your data before printing out a confusion matrix."
        print("accuracy:", accuracy_score(self.Y_test, self.predictions, normalize=True, sample_weight=None))

    def predict(self, input_row):
        """
        Print the result of the classification of the given sample.

        With the given age mapping (instance variable), the returned integer
        from self.predict() serves as a key for the corresponding age group.
        This way we can construct a response in a natural language, telling
        the user the age boundaries of the predicted age group.

        Ex: "You are estimated to be between 50 and 65 years old"

        :param input_row:   array of input values
        :type input_row:    np.ndarray

        :return:            tuple with the result of the classification and a
                            string, telling the user the resulting age group.
        :type return:       tuple(int, str)
        """
        try:
            prediction = int(self.mlp.predict(input_row)[0])
        except ValueError as e:
            return e
        return (prediction, "You are estimated to be between %s and %s years old" % (str(self.age_mapping[prediction][0]), str(self.age_mapping[prediction][1])))
    
    
    def plot_confusion_matrix(self, normalize=True, title='Normalized confusion matrix',
                          cmap=plt.cm.Greys):
        """
        SOURCE: http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting 'normalize=True'.

        :param normalize:   Flag, if the numbers should be normalized
        :type normalize:    bool
        :param title:       title of the resulting plot
        :type title:        str
        :param cmap:        name of the plt.colormap
        :type cmap:         plt.colormap
        """
        classes = self.age_mapping.keys()
        cm = confusion_matrix(self.Y_test,self.predictions)
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        else:
            title = "Confusion matrix, without normalization"

        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel("True label")
        plt.xlabel("Predicted label")
        plt.show()

    def plot_age_group_sizes(self, title='Sizes of age groups used for training', color="black"):
        """
        Plot the number of segments used for training for each age group
        respectively in a single plot (bar chart).

        :param title:       title of the resulting plot
        :type title:        str
        :param color:       color, in which the bars will be plotted
        :type color:        str
        """
        d = {}
        test_data = self.Y_train
        for i in test_data:
            i = int(i)
            try:
                d[i] += 1
            except KeyError:
                d[i] = 1

        plot_values = []
        for j in range(0, max(d.keys())+1):
            plot_values.append(d[j])

        plt.bar(range(len(d)), plot_values, align='center', color=color)
        plt.xticks(range(len(d)), range(0, max(d.keys())+1))
        plt.xlabel("Age groups")
        plt.ylabel("Number of samples for testing")
        plt.title(title)
        plt.show()
