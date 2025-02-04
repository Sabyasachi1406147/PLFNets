# -*- coding: utf-8 -*-
"""
Created on Mon Aug 29 23:03:54 2022

@author: sb3682
"""

import tensorflow as tf
import math
from numpy import log10, roll, linspace
import numpy as np

class SincNetLayer1D(tf.keras.layers.Layer):
    def __init__(self, filter_num, filter_size, sampling_freq, **kwargs):
        self.fnum = filter_num
        self.fsize = filter_size
        self.fs = sampling_freq
        super(SincNetLayer1D, self).__init__(**kwargs)

    def build(self, input_shape):
        # Initialization of the filterbanks in mel scale.
        self.f1_real = self.add_weight(name='filt_b1_real',
                        shape=(self.fnum,),
                        initializer='glorot_uniform',
                        trainable=True)

        self.fbandwidth_real = self.add_weight(name='filt_band_real',
                        shape=(self.fnum,),
                        initializer='glorot_uniform',
                        trainable=True)
        
        self.f1_im = self.add_weight(name='filt_b1_imag',
                        shape=(self.fnum,),
                        initializer='glorot_uniform',
                        trainable=True)

        self.fbandwidth_im = self.add_weight(name='filt_band_imag',
                        shape=(self.fnum,),
                        initializer='glorot_uniform',
                        trainable=True)
        
        # self.f1_im = self.f1_real

        # self.fbandwidth_im = self.fbandwidth_real
        
        mel_low = 10
        mel_high = (2595 * log10(1 + (self.fs / 2) / 700))              # Converting Hz to Mel
        mel_points = linspace(mel_low, mel_high, self.fnum)             # Array of equally spaced frequencies in Mel scale
        freq_points = (700 * (10 ** (mel_points / 2595) - 1))           # Converting Mel back to Hz       
        
        #freq_points = linspace(10,self.fs/2,self.fnum)
        b1_real = roll(freq_points, 1)
        b2_real = roll(freq_points, -1)
        b1_real[0] = 20
        b2_real[-1] = (self.fs / 2)
        b1_im = roll(freq_points, 1)
        b2_im = roll(freq_points, -1)
        b1_im[0] = 20
        b2_im[-1] = (self.fs / 2)
        self.freq_scale = self.fs * 1.0
        self.set_weights([b1_real/self.freq_scale, (b2_real-b1_real)/self.freq_scale, b1_im/self.freq_scale, (b2_im-b1_im)/self.freq_scale])
        #self.set_weights([b1_real/self.freq_scale, (b2_real-b1_real)/self.freq_scale])
        
        super(SincNetLayer1D, self).build(input_shape)  # Be sure to call this at the end
        
    def call(self, input_tensor, **kwargs):
        #########################   real
        min_freq = 20.0
        min_band = 20.0
        self.f1_abs_real = tf.abs(self.f1_real) + min_freq / self.freq_scale
        self.f2_abs_real = self.f1_abs_real + (tf.abs(self.fbandwidth_real) + min_band / self.freq_scale)
        
        self.f1_abs_im = tf.abs(self.f1_im) + min_freq / self.freq_scale
        self.f2_abs_im = self.f1_abs_im + (tf.abs(self.fbandwidth_im) + min_band / self.freq_scale)
        ######################################
        
        # Filter window (hamming).
        n = linspace(0, self.fsize, self.fsize)
        window1 = 0.54 - 0.46 * tf.math.cos(2 * math.pi * n / self.fsize)
        self.window = tf.cast(window1, "float32")
        #window = tf.Variable(initial_value=window2, trainable=True, dtype=tf.float32, name='window')
        
        # Defining the points for sinc function in time domain.
        t_right_linspace = linspace(1, (self.fsize - 1) / 2, int((self.fsize - 1) / 2))
        self.t_right = tf.constant(t_right_linspace / self.fs, dtype=tf.float32, name='t_right')
        # Compute the filters.
        real = input_tensor[:,:,0,tf.newaxis]
        imag = input_tensor[:,:,1,tf.newaxis]
        output_list_real = []
        output_list_im = []
        for i in range(self.fnum):
            low_pass1_real = 2 * self.f1_abs_real[i] * sinc(self.f1_abs_real[i] * self.fs, self.t_right)
            low_pass2_real = 2 * self.f2_abs_real[i] * sinc(self.f2_abs_real[i] * self.fs, self.t_right)
            band_pass_real = (low_pass2_real - low_pass1_real)
            band_pass_real = band_pass_real / tf.math.reduce_max(band_pass_real)
            output_list_real.append(band_pass_real * self.window)
        filters_real = tf.keras.backend.stack(output_list_real)                               # (80,251)
        filters_real = tf.keras.backend.transpose(filters_real)                               # (251,80)
        filters_real = tf.keras.backend.reshape(filters_real, (self.fsize, 1, self.fnum))     # (251,1,80) 
        
        for i in range(self.fnum):
            low_pass1_im = 2 * self.f1_abs_im[i] * sinc(self.f1_abs_im[i] * self.fs, self.t_right)
            low_pass2_im = 2 * self.f2_abs_im[i] * sinc(self.f2_abs_im[i] * self.fs, self.t_right)
            band_pass_im = (low_pass2_im - low_pass1_im)
            band_pass_im = band_pass_im / tf.math.reduce_max(band_pass_im)
            output_list_im.append(band_pass_im * self.window)
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

class ComplexConcatenate(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(ComplexConcatenate, self).__init__(**kwargs)
        
    def build(self, input_shape):
        super(ComplexConcatenate, self).build(input_shape)
        
    def call(self, complex_inputs):
        input1 = complex_inputs[0]
        input2 = complex_inputs[1]
        input3 = complex_inputs[2]
        input1_shape = input1.shape
        real_input1 = input1[:,:,:int(input1_shape[2]/2)]
        imag_input1 = input1[:,:,int(input1_shape[2]/2):]
        input2_shape = input2.shape
        real_input2 = input2[:,:,:int(input2_shape[2]/2)]
        imag_input2 = input2[:,:,int(input2_shape[2]/2):]
        input3_shape = input3.shape
        real_input3 = input3[:,:,:int(input3_shape[2]/2)]
        imag_input3 = input3[:,:,int(input3_shape[2]/2):]
        
        output = tf.concat([real_input1, real_input2, real_input3, imag_input1, imag_input2, imag_input3], axis=2)
        return output