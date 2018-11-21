#!/usr/bin/python
#-*- coding: utf-8 -*-
# Author: Linus Manser, 13-791-132
# date: 20.10.2018
# Additional Info: python3 (3.6.2) program

import os
import librosa
import numpy as np
import scipy
import sidekit
from sidekit.frontend import features
from sidekit.frontend.vad import *

PARAM_TYPE = np.float64


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
    mfcc_vals = sidekit.mfcc(y)[0]
    
    # mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N)
    # mfcc = np.transpose(mfcc)
    d = 2
    p = 3
    k = 2
    sdc = sidekit.shifted_delta_cepstral(mfcc_vals, d=d, p=p, k=k)
    print("shifted_delta_cepstral(mfcc, d=%i, p=%i, k=%i)" % (d,p,k))
    print("shape mfcc:", mfcc_vals.shape)
    print(mfcc_vals)
    print("shape SDC", sdc.shape)
    print(sdc)
    
if __name__ == '__main__':
    main()


