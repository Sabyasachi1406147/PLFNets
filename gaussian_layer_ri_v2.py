# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 23:29:33 2024

@author: sb3682
"""

import tensorflow as tf
import math
from numpy import log10, roll, linspace
import numpy as np

class FcConstraint(tf.keras.constraints.Constraint):
    def __init__(self, fs):
        self.fs = fs

    def __call__(self, w):
        return tf.clip_by_value(w, -self.fs / 2, self.fs / 2)
    
class BConstraint(tf.keras.constraints.Constraint):
    def __init__(self, fs):
        self.fs = fs

    def __call__(self, w):
        return tf.clip_by_value(w, 1e-3, self.fs / 8)  # Avoid zero bandwidt

class GaussNetLayer1D(tf.keras.layers.Layer):
    def __init__(self, filter_num, filter_size, sampling_freq, **kwargs):
        self.fnum = filter_num
        self.fsize = filter_size
        self.fs = sampling_freq
        super(GaussNetLayer1D, self).__init__(**kwargs)

    def build(self, input_shape):
        # Initialization of the filter banks in mel scale.
        self.fc = self.add_weight(name='filt_fc',
                     shape=(self.fnum,),
                     initializer='glorot_uniform',
                     trainable=True,
                     constraint=FcConstraint(self.fs))
         
        self.fbandwidth = self.add_weight(name='filt_band',
                         shape=(self.fnum,),
                         initializer=tf.keras.initializers.RandomUniform(minval=1e-3,maxval=self.fs/8),  # Small positive initial value
                         trainable=True,
                         regularizer=tf.keras.regularizers.l2(1e-4),  # Small L2 regularization
                         constraint=BConstraint(self.fs))
         
        # Mel scale conversion
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
        self.freq_scale = self.fs * 1.0
         
        self.set_weights([b1 / self.freq_scale, (b2 - b1) / (2 * self.freq_scale)])
         
        super(GaussNetLayer1D, self).build(input_shape)

    def call(self, input_tensor, **kwargs):
        # Hamming window
        n = linspace(0, self.fsize, self.fsize, dtype=np.float32)
        window1 = 0.54 - 0.46 * tf.math.cos(2 * math.pi * n / self.fsize)
        window = tf.cast(window1, "float32")
        window = window / tf.reduce_max(window)

        # Defining the points for the Gaussian function in the time domain
        t_linspace = linspace(1, self.fsize, int(self.fsize), dtype=np.float32)
        self.t = tf.constant(t_linspace / self.fs, dtype=tf.float32, name='t')

        # Compute the filters
        real = input_tensor[:, :, 0, tf.newaxis]
        imag = input_tensor[:, :, 1, tf.newaxis]
        output_list_real = []
        output_list_im = []
        for i in range(self.fnum):
            temp = gauss(self.fbandwidth[i] * self.fs, self.t)
            # temp = temp * window
            temp_real = temp * tf.math.cos(2 * math.pi * self.fc[i] * n)
            temp_imag = temp * tf.math.sin(2 * math.pi * self.fc[i] * n)
            output_list_real.append(temp_real)
            output_list_im.append(temp_imag)
               
        filters_real = tf.keras.backend.stack(output_list_real)  # (80, 251)
        filters_real = tf.keras.backend.transpose(filters_real)  # (251, 80)
        filters_real = tf.keras.backend.reshape(filters_real, (self.fsize, 1, self.fnum))  # (251, 1, 80)

        filters_im = tf.keras.backend.stack(output_list_im)  # (80, 251)
        filters_im = tf.keras.backend.transpose(filters_im)  # (251, 80)
        filters_im = tf.keras.backend.reshape(filters_im, (self.fsize, 1, self.fnum))  # (251, 1, 80)
         
        # Do the convolution
        out_rr = tf.keras.backend.conv1d(real, kernel=filters_real)
        out_ii = tf.keras.backend.conv1d(imag, kernel=filters_im)
        out_ri = tf.keras.backend.conv1d(real, kernel=filters_im)
        out_ir = tf.keras.backend.conv1d(imag, kernel=filters_real)
         
        out_real = out_rr - out_ii
        out_imag = out_ri + out_ir
         
        out = tf.concat([out_real, out_imag], 2)
        return out

    def compute_output_shape(self, input_shape):
        new_size = tf.python.keras.utils.conv_utils.conv_output_length(input_shape[1], self.fsize, 
                                        padding="valid", stride=1, dilation=1)
        return (input_shape[0],) + (new_size,) + (self.fnum,)

def gauss(band, t):
    epsilon = 1e-10  # Small number to prevent division by zero
    band = tf.maximum(band, epsilon)
    sigma = tf.math.sqrt(tf.math.log(2.0)) / (math.pi * band)
    gauss = tf.math.exp((-t ** 2) / (2 * (sigma ** 2))) * tf.math.sqrt(2 / (tf.math.sqrt(math.pi) * sigma))
    gauss = gauss / tf.reduce_max(gauss)
    return gauss
