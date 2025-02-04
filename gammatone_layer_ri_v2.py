# -*- coding: utf-8 -*-
"""
Created on Sat Jul 13 12:15:06 2024

@author: sb3682
"""

import tensorflow as tf
import math
from numpy import log10, roll, linspace
import numpy as np
import pdb

class GammatoneNetLayer1D(tf.keras.layers.Layer):
    def __init__(self, filter_num, filter_size, sampling_freq, **kwargs):
        self.fnum = filter_num
        self.fsize = filter_size
        self.fs = sampling_freq
        super(GammatoneNetLayer1D, self).__init__(**kwargs)

    def build(self, input_shape):
        # Initialization of the filterbanks in mel scale.
        self.fc = self.add_weight(name='filt_fc',
                        shape=(self.fnum,),
                        initializer='glorot_uniform',
                        trainable=True)

        self.fbandwidth = self.add_weight(name='filt_band',
                        shape=(self.fnum,),
                        initializer='glorot_uniform',
                        trainable=True)

        self.forder = self.add_weight(name='filt_order',
                        shape=(self.fnum,),
                        initializer='glorot_uniform',
                        trainable=True,
                        constraint=lambda x: tf.clip_by_value(tf.round(x), 1, 9))
        
        mel_low = 10
        mel_high = (2595 * log10(1 + (self.fs / 2) / 700))  # Converting Hz to Mel
        mel_points = linspace(mel_low, mel_high, self.fnum // 2)  # Array of equally spaced frequencies in Mel scale
        freq_points_positive = (700 * (10 ** (mel_points / 2595) - 1))
        freq_points_negative = -(700 * (10 ** (mel_points / 2595) - 1))  # Converting Mel back to Hz
        freq_points_negative = np.flip(freq_points_negative, axis=[0])
        freq_points = np.concatenate((freq_points_negative, freq_points_positive), axis=0)
        b1 = roll(freq_points, 1)
        b2 = roll(freq_points, -1)
        b1[0] = b1[1] - 20
        b2[-1] = b2[-2] + 20
        b3 = np.random.randint(1,10,256)
        # pdb.set_trace()
        self.freq_scale = self.fs * 1.0
        self.set_weights([b1 / self.freq_scale, (b2 - b1) / (2 * self.freq_scale), b3])
        
        super(GammatoneNetLayer1D, self).build(input_shape)  # Be sure to call this at the end
        
    def call(self, input_tensor, **kwargs):
        
        # Filter window (hamming).
        n = linspace(0, self.fsize, self.fsize)
        window1 = 0.54 - 0.46 * tf.math.cos(2 * math.pi * n / self.fsize)
        window = tf.cast(window1, "float32")
        window = window/tf.reduce_max(window)
        
        # Defining the points for sinc function in time domain.
        t_linspace = linspace(1, self.fsize, int((self.fsize)))
        self.t = tf.constant(t_linspace / self.fs, dtype=tf.float32, name='t')
        # print(self.fsize)
        # Compute the filters.
        real = input_tensor[:,:,0,tf.newaxis]
        imag = input_tensor[:,:,1,tf.newaxis]
        output_list_real = []
        output_list_im = []
        for i in range(self.fnum):
            temp = gammatone(self.fbandwidth[i]*self.fs,self.t,self.forder[i])
            # print(temp.shape)
            temp = temp*window
            temp_real = temp*tf.math.cos(2*math.pi*self.fc[i]*n)
            temp_imag = temp*tf.math.sin(2*math.pi*self.fc[i]*n)
            output_list_real.append(temp_real)
            output_list_im.append(temp_imag)
            
            
        filters_real = tf.keras.backend.stack(output_list_real)                               # (80,251)
        filters_real = tf.keras.backend.transpose(filters_real)                               # (251,80)
        filters_real = tf.keras.backend.reshape(filters_real, (self.fsize, 1, self.fnum))     # (251,1,80) 

        filters_im = tf.keras.backend.stack(output_list_im)                               # (80,251)
        filters_im = tf.keras.backend.transpose(filters_im)                               # (251,80)
        filters_im = tf.keras.backend.reshape(filters_im, (self.fsize, 1, self.fnum))     # (251,1,80) 
        # Do the convolution.
        out_rr = tf.keras.backend.conv1d(real, kernel=filters_real)
        out_ii = tf.keras.backend.conv1d(imag, kernel=filters_im)
        out_ri = tf.keras.backend.conv1d(real, kernel=filters_im)
        out_ir = tf.keras.backend.conv1d(imag, kernel=filters_real)
        
        out_real = (out_rr-out_ii)
        out_imag = (out_ri+out_ir)
        # out_real = out_rr
        # out_imag = out_ir
        
        out = tf.concat([out_real, out_imag], 2)
        return out

    def compute_output_shape(self, input_shape):
        new_size = tf.python.keras.utils.conv_utils.conv_output_length(input_shape[1], self.fsize, 
                                                                        padding="valid", stride=1, dilation=1)
        return (input_shape[0],) + (new_size,) + (self.fnum,)

def gammatone(band, t, n):
    epsilon = 1e-10  # Small number to prevent division by zero
    band = tf.abs(band)
    band = tf.maximum(band, epsilon)
    n = 4  # filter order
    sigma = ((band / 2) * tf.math.sqrt(1 / (2 ** (1 / n) - 1)))
    gammatone = (t ** (n - 1)) * tf.math.exp(-2 * math.pi * sigma * t)
    gammatone = gammatone / tf.reduce_max(gammatone)
    return gammatone
