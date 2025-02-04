
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 13 12:15:06 2024

@author: sb3682
"""

import tensorflow as tf
import math
from numpy import log10, linspace
import numpy as np

class GaussianNetLayer1D(tf.keras.layers.Layer):
    def __init__(self, filter_num, filter_size, sampling_freq, **kwargs):
        super(GaussianNetLayer1D, self).__init__(**kwargs)
        self.fnum = filter_num
        self.fsize = filter_size
        self.fs = sampling_freq

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
        
        mel_low = 10
        mel_high = 2595 * log10(1 + (self.fs / 2) / 700)  # Converting Hz to Mel
        mel_points = linspace(mel_low, mel_high, self.fnum // 2)  # Array of equally spaced frequencies in Mel scale
        freq_points_positive = 700 * (10 ** (mel_points / 2595) - 1)
        freq_points_negative = -freq_points_positive[::-1]  # Converting Mel back to Hz
        freq_points = np.concatenate((freq_points_negative, freq_points_positive), axis=0)
        b1 = np.roll(freq_points, 1)
        b2 = np.roll(freq_points, -1)
        b1[0] = b1[1] - 20
        b2[-1] = b2[-2] + 20
        self.freq_scale = self.fs * 1.0
        self.set_weights([b1 / self.freq_scale, (b2 - b1) / (2 * self.freq_scale)])
        
        self.n = tf.constant(linspace(0, self.fsize, self.fsize), dtype=tf.float32)
        self.window = 0.54 - 0.46 * tf.math.cos(2 * math.pi * self.n / self.fsize)
        self.t = tf.constant(linspace(1, self.fsize, self.fsize) / self.fs, dtype=tf.float32, name='t')
        
        super(GaussianNetLayer1D, self).build(input_shape)  # Be sure to call this at the end

    def call(self, input_tensor, **kwargs):
        # Defining the points for sinc function in time domain.
        window = tf.cast(self.window, "float32")
        
        # Vectorized filter computation
        t = self.t[:, tf.newaxis]
        fc = tf.reshape(self.fc, [1, self.fnum])
        fbandwidth = tf.reshape(self.fbandwidth, [1, self.fnum])
        
        windowed_filter = gauss(fbandwidth*self.fs,self.t) * window[:, tf.newaxis]
        
        temp_real = windowed_filter * tf.math.cos(2 * math.pi * fc * self.n[:, tf.newaxis])
        
        filters = tf.transpose(temp_real, perm=[1, 0])
        filters = tf.reshape(filters, [self.fnum, self.fsize, 1])
        
        out = tf.keras.backend.conv1d(input_tensor, kernel=filters)
        return out

    def compute_output_shape(self, input_shape):
        new_size = tf.keras.utils.conv_utils.conv_output_length(input_shape[1], self.fsize, 
                                                                padding="valid", stride=1, dilation=1)
        return (input_shape[0], new_size, self.fnum)

def gauss(band, t):
    sigma = tf.math.sqrt(tf.math.log(2.0))/ (2*math.pi*band)
    gauss = tf.math.exp((-t**2)/2*(sigma**2))/(tf.math.sqrt(2*math.pi*sigma))
    return gauss