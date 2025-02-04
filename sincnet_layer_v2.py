# -*- coding: utf-8 -*-
"""
Created on Mon Aug 29 23:03:54 2022

@author: sb3682
"""

import tensorflow as tf
import math
from numpy import log10, roll, linspace
import numpy as np
import pdb

class SincNetLayer1D(tf.keras.layers.Layer):
    def __init__(self, filter_num, filter_size, sampling_freq, **kwargs):
        self.fnum = filter_num
        self.fsize = filter_size
        self.fs = sampling_freq
        super(SincNetLayer1D, self).__init__(**kwargs)

    def build(self, input_shape):
        # Initialization of the filterbanks in mel scale.
        self.f1 = self.add_weight(name='filt_b1',
                        shape=(self.fnum,),
                        initializer='uniform',
                        trainable=True)

        self.fbandwidth = self.add_weight(name='filt_band',
                        shape=(self.fnum,),
                        initializer='uniform',
                        trainable=True)
        mel_low = 80
        mel_high = (2595 * log10(1 + (self.fs / 2) / 700))              # Converting Hz to Mel
        mel_points = linspace(mel_low, mel_high, self.fnum)             # Array of equally spaced frequencies in Mel scale
        freq_points = (700 * (10 ** (mel_points / 2595) - 1))           # Converting Mel back to Hz
        
        #freq_points = linspace(20,self.fs/2,self.fnum)
        b1 = roll(freq_points, 1)
        b2 = roll(freq_points, -1)
        b1[0] = 20
        b2[-1] = (self.fs / 2) - 100
        self.freq_scale = self.fs * 1.0
        self.set_weights([b1/self.freq_scale, (b2-b1)/self.freq_scale])
        
        
        super(SincNetLayer1D, self).build(input_shape)  # Be sure to call this at the end
        
    def call(self, input_tensor, **kwargs):
        #########################   real
        min_freq = 20.0
        min_band = 40.0
        self.f1_abs = tf.abs(self.f1) + min_freq / self.freq_scale
        self.f2_abs = self.f1_abs + (tf.abs(self.fbandwidth) + min_band / self.freq_scale)
        
        ######################################
        
        # Filter window (hamming).
        n = linspace(0, self.fsize, self.fsize)
        window1 = 0.54 - 0.46 * tf.math.cos(2 * math.pi * n / self.fsize)
        self.window = tf.cast(window1, "float32")
        #window = tf.Variable(initial_value=window2, trainable=True, dtype=tf.float32, name='window')
        
        # Defining the points for sinc function in time domain.
        t_right_linspace = linspace(1, (self.fsize - 1) / 2, int((self.fsize - 1) / 2))
        self.t_right = tf.constant(t_right_linspace / self.fs, dtype=tf.float32, name='t_right')
        #self.t_right = tf.Variable(initial_value=t_right_linspace/self.fs, trainable=True, dtype=tf.float32, name='t_right')
        # Compute the filters.
        output_list = []
        for i in range(self.fnum):
            low_pass1 = 2 * self.f1_abs[i] * sinc(self.f1_abs[i] * self.fs, self.t_right)
            low_pass2 = 2 * self.f2_abs[i] * sinc(self.f2_abs[i] * self.fs, self.t_right)
            band_pass = (low_pass2 - low_pass1)
            band_pass = band_pass / tf.math.reduce_max(band_pass)
            output_list.append(band_pass * self.window)
        filters = tf.keras.backend.stack(output_list)                               # (80,251)
        filters = tf.keras.backend.transpose(filters)                               # (251,80)
        filters = tf.keras.backend.reshape(filters, (self.fsize, 1, self.fnum))     # (251,1,80) 
        # Do the convolution.
        out = tf.keras.backend.conv1d(input_tensor, kernel=filters)
        return out

    def compute_output_shape(self, input_shape):
        new_size = tf.python.keras.utils.conv_utils.conv_output_length(input_shape[1], self.fsize, 
                                                                       padding="valid", stride=1, dilation=1)
        return (input_shape[0],) + (new_size,) + (self.fnum,)
"""
    # Overriding get_config method as __init__ function has positional arguements
    def get_config(self):
        return {"Number of filters": self.fnum,
                "Filter size": self.fsize,
                "Sampling Frequency":self.fs}

    @classmethod
    def from_config(cls, config):
        return cls(**config)
"""
def sinc(band, t_right):
    y_right = tf.math.sin(2 * math.pi * band * t_right) / (2 * math.pi * band * t_right)
    y_left = tf.reverse(y_right, axis=[0])
    return tf.concat([y_left, tf.constant([1], dtype=tf.float32), y_right], axis=0)