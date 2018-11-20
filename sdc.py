#!/usr/bin/python
#-*- coding: utf-8 -*-
# Author: Linus Manser, 13-791-132
# date: 20.10.2018
# Additional Info: python3 (3.6.2) program

import os
import librosa
import numpy as np

def shifted_delta_cepstral(cep, d=5, p=2, k=2):
    # taken from:
    # https://projets-lium.univ-lemans.fr/sidekit/_modules/frontend/features.html
    """
    Compute the Shifted-Delta-Cepstral features for language identification

    :param cep: matrix of feature, 1 vector per line
    :param d: represents the time advance and delay for the delta computation
    :param k: number of delta-cepstral blocks whose delta-cepstral
       coefficients are stacked to form the final feature vector
    :param p: time shift between consecutive blocks.

    return: cepstral coefficient concatenated with shifted deltas
    """

    y = np.r_[np.resize(cep[0, :], (d, cep.shape[1])),
                 cep,
                 np.resize(cep[-1, :], (k * 3 + d, cep.shape[1]))]

    delta = compute_delta(y, win=d, method='diff')
    sdc = np.empty((cep.shape[0], cep.shape[1] * k))

    idx = np.zeros(delta.shape[0], dtype='bool')
    for ii in range(k):
        idx[d + ii * p] = True
    for ff in range(len(cep)):
        sdc[ff, :] = delta[idx, :].reshape(1, -1)
        idx = np.roll(idx, 1)
    return np.hstack((cep, sdc))


def compute_delta(features,
                  win=3,
                  method='filter',
                  filt=np.array([.25, .5, .25, 0, -.25, -.5, -.25])):
    # taken from:
    # https://projets-lium.univ-lemans.fr/sidekit/_modules/frontend/features.html
    """features is a 2D-ndarray  each row of features is a a frame

    :param features: the feature frames to compute the delta coefficients
    :param win: parameter that set the length of the computation window.
            The size of the window is (win x 2) + 1
    :param method: method used to compute the delta coefficients
        can be diff or filter
    :param filt: definition of the filter to use in "filter" mode, default one
        is similar to SPRO4:  filt=np.array([.2, .1, 0, -.1, -.2])

    :return: the delta coefficients computed on the original features.
    """
    # First and last features are appended to the begining and the end of the
    # stream to avoid border effect
    PARAM_TYPE = np.float64
    x = np.zeros((features.shape[0] + 2 * win, features.shape[1]), dtype=PARAM_TYPE)
    x[:win, :] = features[0, :]
    x[win:-win, :] = features
    x[-win:, :] = features[-1, :]

    delta = np.zeros(x.shape, dtype=PARAM_TYPE)

    if method == 'diff':
        filt = np.zeros(2 * win + 1, dtype=PARAM_TYPE)
        filt[0] = -1
        filt[-1] = 1

    for i in range(features.shape[1]):
        delta[:, i] = np.convolve(features[:, i], filt)

    return delta[win:-win, :]


def sdc(mfcc):
    """
    """
    data = mfcc # make a local copy in order to be able to apply changes
    number_of_blocks = data[0,:].size # number of "frames"
    N = data[:,0].size # number of c coefficients
    d = 2 # This will result in a "convolution window" of 5 rows
    step_size = 2*d+1
    # with this step_size, I am ensuring that every row has been visited once
    while number_of_blocks > step_size: 
        # continue to compute delta coefficients until you don't have enough
        # blocks to continue anymore.
        sdc_data = []
        for step in range(d, number_of_blocks-d, step_size):
            # make steps over the array. each step has a minumum size 
            # of the convoluting window defined by the spread (-d...d)
            # This way, I can gather the first x rows of the array and compute
            # the delta coefficients values for these x rows. After that, I can 
            # step forward, so that I am at the middle of the next "convoluting window"
            delta_data = []
            for coeff in range(0, N):
                # calculate delta values for each coefficient
                # tried to do this according to:
                # https://www.researchgate.net/publication/221478246_Selection_of_the_best_set_of_shifted_delta_cepstral_features_in_speaker_verification_using_mutual_information
                # NB: I don't really know if my algorithm is correct, since hd (in formula 1) is not defined in the paper and I don't know what hd is
                upper = 0
                lower = 0
                for j in range(-d, d+1):
                    # include all vectors within the spread and sum them up as defined 
                    # TODO: check if j*data[i, step+j] is correct (What is hd?, at the moment its missing)
                    try:
                        # the upper/lower should correspong to the formula given in the paper
                        # upper = above the division line, lower = below the line
                        upper += j*data[coeff, step+j]
                        lower += j**2
                    except IndexError:
                        # in case of an indexError, the index is too large
                        # take a value within the same convoluting window and
                        # duplicate the values by this. The resulting error
                        # is not as bad any other value
                        upper += j*data[coeff, 0]
                        lower += j**2
                r = upper/lower
                # the summarized values are appended to the delta_data for each window
                delta_data.append(r)    
                
            sdc_data.append(delta_data)
        # bring in correct form by transposing the matrix
        data = np.array(sdc_data).transpose()
        number_of_blocks = data[0,:].size
        
    return data
    
def combine_data_rows(data):
    """
    at this moment, this function takes the average of the remaining SDC rows.
    This will always result in a single return value for each dimension.
    """
    rows, cols = data.shape
    out_data = []
    for r in range(0, rows):
        tmp = []
        for c in range(0, cols):
            tmp.append(data[r,c])
        # for now: just take the mean of the remaining columns in order to 
        # reduce the dimensions of this array from (N, x) to (N, 1) where N is the
        # number of coefficients (39 in our cases)
        out_data.append(np.mean(tmp))
    return out_data
    

def main():
    p = "/Users/linusmanser/Desktop/3sec.wav"
    sampling_rate = 16000
    N = 13
    y, sr = librosa.load(p, sampling_rate)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N)
    mfcc = np.transpose(mfcc)
    print("M", mfcc.shape)
    data = shifted_delta_cepstral(mfcc)
    print("D", data.shape)
    print(data)
    # data = sdc(M)
    # data = combine_data_rows(data)
    # print(data)
    # testdata = np.random.random([1,39])
    # print(testdata)
    # testrun = sdc(testdata)
    # print(testrun)
    
if __name__ == '__main__':
    main()


