#!/usr/bin/python
#-*- coding: utf-8 -*-
# Author: Linus Manser, 13-791-132
# date: 07.12.2018
# Additional Info: python3 (3.6.2) program
#                  Collection of Classes used for this project

import parselmouth
import itertools
import subprocess
import os
import numpy as np
import pandas as pd
from shutil import rmtree
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import librosa
import sidekit
import time
from joblib import dump, load


class CSVhandler(object):
    def __init__(self, csv_filename):
        self.csv_filename = csv_filename
        self.df = self.cast(pd.read_csv(self.csv_filename), float)
        
    def cast(self, df, type):
        """
        casts and cruedly cleans a panda dataframe into the given type

        :param df:      input dataframe to be casted
        :type df:       pd.DataFrame
        :param type:    type of casted pandaframe values
        :type type:     type
        :return:        casted dataframe
        :type return:   pd.DataFrame
        """
        try:
            df = df.astype(type)
        except ValueError:
            for n in list(df):
                try:
                    l = df.loc[df[n].map(str)=="--undefined--"].index
                    df.drop(l, inplace=True)
                except KeyError:
                    pass
            df = df.astype(type)
        return df
            
    def apply_sdc(self, measurement, indexmap=None, mode="mean"):
        """
        applies shifted-delta-coefficients to a given collection of frame data.
        SDC convolutes longitudinal data into a reduced stack of dimensions

        :param measurement: extracted measurement values of a feature
        :type measurement:  pd.DataFrame
        :param indexmap:    map that allows for linking of the newly created
                            feature vectors
        :type indexmap:     dict
        :param mode:        describes how the remaining values are convoluted
                            into the one-dimensional vector
        :type mode:         str
        :return:            reduced and convoluted shifted delta coefficients
                            of the given input measurements
        :type return:       nd.array
        """
        data = self.df[measurement].values
        data = np.reshape(data, (-1, 1))
        sdc = sidekit.shifted_delta_cepstral(data, d=2, p=3, k=7)
        if mode == "mean":
            sdc_data = np.mean(sdc, axis=0).tolist()
        # in case variance is what I want to extract as well, I could do smth here
        elif mode == "variance":
            sdc_data = np.var(sdc, axis=0).tolist()
        # update indexmap in order to find the correct columns later on
        cols = len(sdc_data)
        if indexmap:
            first_index = max(map(max, indexmap.values())) + 1
            indexmap[measurement] = [i for i in range(first_index, first_index + cols)]
        return sdc_data
    
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
    
    def extract_features(self, interval, outfile_path, SEGMENT_PATH, EXTRACTION_PATH, PROGRESS_PATH, PROGRESS_FILE_PATH, processed_recordings=[]):
        """
        Extract features from a given SpeechRecording by segmenting it into
        multiple segments and running the praat script on each segment.

        :param interval:        duration of each segment (in seconds)
        :type interval:         int
        :param outfile_path:    path to the output file
        :type outfile_path:     str
        :param SEGMENT_PATH:    path to the segment folder
        :type SEGMENT_PATH:     str
        :param EXTRACTION_PATH: path to the extraction folder
        :type EXTRACTION_PATH:  str
        :param PROGRESS_PATH:   path to the progress folder
        :type PROGRESS_PATH:    str
        :param PROGRESS_FILE_PATH:  path to progress file
        :type PROGRESS_FILE_PATH:   str
        :param processed_recordings: list of already processed recordings
        :type processed_recordings:  list

        :return:                header corresponding to the extracted feature
                                values
        :type return:           str
        """
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
        # if resume-flag is not given, delete the existing "progress" status
        # and start with a new one
        indexmap = {}
        segments = self.segment_wave(interval, SEGMENT_PATH)
        all_vals = []
        current_pp = ""
        header = None
        # gather all data and set up correct table for each segment
        for identification, age, gender, p, pp in segments:
            if pp in processed_recordings:
                return
            else:
                current_pp = pp
            try:
                subprocess.check_call(['/Applications/Praat.app/Contents/MacOS/Praat', '--run', 'code/featureExtraction.praat', '-25', '2', '0.3', '0', SEGMENT_PATH, EXTRACTION_PATH, str(identification)])
            except subprocess.CalledProcessError:
                # discard segment if subprocess throws errors due to invalid data
                print(" ! Segment not processed, due to invalid values -> Problem in '...%s'" % pp[-10:])
                continue
            endings = ["_formanttable.csv", "_harmtable.csv", "_pitchtable.csv", "_rest.csv", "_spectable.csv"]
            # converting strings to number in order to better handle the dataframes
            if gender == "m" or gender == "male":
                gender = 0
            else:
                gender = 1
            indexmap = {"age":[0], "gender":[1]}
            ignore_list = ["age", "gender"]
            feature_values = [age, gender]
            # calculate SDCs via MFCCs
            y, sr = librosa.load(p)
            mfcc = sidekit.mfcc(y)[0]
            mfcc_delta = sidekit.compute_delta(mfcc)
            mfcc_delta_delta = sidekit.compute_delta(mfcc_delta)
            M = np.concatenate((mfcc,mfcc_delta,mfcc_delta_delta), axis=1)
            # computing SDCs with the given parameter
            # X-1-3-7
            sdc = sidekit.shifted_delta_cepstral(M, d=1, p=3, k=7)
            sdc_data_mean = np.mean(sdc, axis=0).tolist()
            # also possible to include the variance matrix
            # sdc_data_var = np.var(sdc, axis=0).tolist()
            indexmap["mfcc_mean"] = [i for i in range(2,2+len(sdc_data_mean))]
            feature_values += sdc_data_mean
            # extract and add the other features to the segmental feature data
            # procedure for ending "_rest.csv"
            for ending in endings:
                d = CSVhandler(EXTRACTION_PATH + str(identification) + ending)
                if ending != "_rest.csv":
                    for f in list(d.df):
                        feature_values += d.apply_sdc(f, indexmap=indexmap)
                else:
                    for f in list(d.df):
                        ignore_list.append(f)
                        if f != "soundname":
                            indexmap[f] = [0]
                            feature_values.append(d.df[f].values[0])
            header = make_header(indexmap, ignore_list)
            all_vals.append(feature_values)
        
        with open(outfile_path, "a+") as out_file:
            for line in all_vals:
                line = ",".join(map(str, line)) + "\n"
                out_file.write(line)

        # add the processed recording to the progress file
        with open(PROGRESS_FILE_PATH, "a+") as prg_file:
            prg_file.write(current_pp + "\n")

        return header
        
    def segment_wave(self, interval, SEGMENT_PATH):
        """
        Segment the input signal into segments of given length

        :param interval:        duration of segments in seconds
        :type interval:         int
        :param SEGMENT_PATH:    path to the segment folder, where the segments
                                will be saved to
        :type SEGMENT_PATH:     str
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
    to get segments over which the feature extraction will run.

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
    AgeClassifier class. Creates a classifier instance with the given data.
    The created classifier is a Multi-Layer Perceptron (MLP) Classifier.
    Depending on whether a modelname (no model loaded) or model (model given)
    is given, the initialisation of the instance is different. If a model is
    given, it will be used to fit the classifier. If only a modelname is given,
    the AgeClassifier instance will create and save a new model.

    :param root_path:           path to the root of the system, since it relies
                                on the existence of training and testing or
                                model files.
    :type root_path:            str
    :param age_mapping:         age mapping
    :type age_mapping:          pd.DataFrame
    :param modelname:           if no model was given, calculate a new one with this name
    :type modelname:            str
    :param hidden_layer_sizes:  tuple of hidden layer sizes.
    :type hidden_layer_sizes:   tuple
    :param activation:          activation function used in the nodes of the NN
    :type activation:           str
    :param max_iter:            maximum number of fitting iterations
    :type max_iter:             int
    :param model:               name of the model that is used for classification
    :type model:                str
    :param features:            list of feature names
    :type features:             list
    :param gender:              gender tag ("m" or "f")
    :type gender:               str
    """
    def __init__(self, root_path, age_mapping, modelname=None, hidden_layer_sizes=(30,30,30), activation='logistic', max_iter=1000, model=None, features=[], gender="m"):
        self.root_path = root_path
        self.age_mapping = age_mapping
        self.features = features
        if self.features != []:
            if gender == "m":
                self.train_df = pd.read_csv(self.root_path + "appdata/train/m_train.csv")
                self.test_df = pd.read_csv(self.root_path + "appdata/test/m_test.csv")
            else:
                self.train_df = pd.read_csv(self.root_path + "appdata/train/f_train.csv")
                self.test_df = pd.read_csv(self.root_path + "appdata/test/f_test.csv")
            self.train_df = self.select_features(self.train_df)
            self.test_df = self.select_features(self.test_df)
            self.X_train, self.Y_train = self.split_target(self.train_df)
            self.X_test, self.Y_test = self.split_target(self.test_df)
            self.predictions = []
            self.scaler = StandardScaler()
            self.scaler.fit(self.X_train)
            self.X_train = self.scaler.transform(self.X_train)
            self.X_test = self.scaler.transform(self.X_test)
        self.modelname = modelname
        self.model = model

        if self.model:
            print(" + model given, computing predictions with given model")
            self.mlp = load(self.model)
        elif not self.model and self.modelname:
            print(" + no predefined model, fitting a new one with name '%s'" % self.modelname)
            self.mlp = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes,activation=activation,max_iter=max_iter)
            self.mlp.fit(self.X_train, self.Y_train)
            try:
                os.mkdir("appdata/models/")
            except FileExistsError:
                pass
            dump(self.mlp, "appdata/models/" + self.modelname)
            print(" + dumped new model named '%s'" % self.modelname)
        else:
            print(" + either modelname nor model was given, program terminated")
            exit()

        if self.features != []:
            self.predictions = self.mlp.predict(self.X_test)

    def select_features(self, df):
        """
        Returns feature values of the created classifier

        :param df:      dataframe, from which the feature values will be taken
        :type df:       pd.DataFrame
        :return:        filtered dataframe
        :type return:   pd.DataFrame
        """
        feats = self.features + ["age", "gender", "age_class"]
        return df.loc[:,feats]

    def split_target(self, df):
        """
        Splice the target values from the other feature values. Namely age_class.

        :param df:      input dataframe, from which the data will be taken
        :type df:       pd.DataFrame
        :return:        tuple(training data, target data)
        :type return:   (np.ndarray, np.ndarray)
        """
        df = df.dropna()
        target = df["age_class"]
        df = df.drop(["age", "age_class","gender"], axis=1)
        return df.as_matrix(), target.as_matrix()

    def print_accuracy_score(self):
        """
        Return a accuracy score of the Classifier with the given training/test data.

        :return:            string stating the accuracy of self
        """
        if self.predictions == []:
            return "you must set up your data before printing out a confusion matrix."
        acc = accuracy_score(self.Y_test, self.predictions, normalize=True, sample_weight=None)
        print("accuracy:", acc)
        return acc

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
        return (prediction, "You are estimated to be between %s and %s years old" % (str(self.age_mapping.loc[prediction, "lowerbound"]), str(self.age_mapping.loc[prediction, "upperbound"])))

    def plot_classification_report(self):
        """
        SOURCE: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html#sklearn.metrics.classification_report
        This function prints and plots a classification report.
        """
        cr = classification_report(self.Y_test,self.predictions)
        print("classification report for %s:" % self.model)
        print(cr)
    
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
        # quick hack of classes, can be improved
        classes = [0,1,2,3,4]
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