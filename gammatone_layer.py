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
        mel_low = 40
        mel_high = (2595 * log10(1 + (self.fs / 2) / 700))  # Converting Hz to Mel
        mel_points = linspace(mel_low, mel_high, self.fnum)  # Array of equally spaced frequencies in Mel scale
        freq_points = (700 * (10 ** (mel_points / 2595) - 1)) 
        b1 = roll(freq_points, 1)
        b2 = roll(freq_points, -1)
        b1[0] = b1[1] - 20
        b2[-1] = b2[-2] + 20
        self.freq_scale = self.fs * 1.0
        self.set_weights([b1/self.freq_scale, (b2-b1)/(2*self.freq_scale)])
        
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
        
        # Compute the filters.
        output_list = []
        for i in range(self.fnum):
            temp = gammatone(self.fbandwidth[i] * self.fs, self.t)
            temp = temp * window
            temp_real = temp * tf.math.cos(2 * math.pi * self.fc[i] * n)
            output_list.append(temp_real)
            
        filters_real = tf.keras.backend.stack(output_list)  # (80,251)
        filters_real = tf.keras.backend.transpose(filters_real)  # (251,80)
        filters_real = tf.keras.backend.reshape(filters_real, (self.fsize, 1, self.fnum))  # (251,1,80) 
        
        # # Check for NaNs in filters before convolution
        # tf.debugging.check_numerics(filters_real, "filters_real contains NaNs or Infs") 
        
        # Do the convolution.
        out = tf.keras.backend.conv1d(input_tensor, kernel=filters_real)
        
        # Check for NaNs in output
        # tf.debugging.check_numerics(out, "Output contains NaNs or Infs")
        
        return out

    def compute_output_shape(self, input_shape):
        new_size = tf.python.keras.utils.conv_utils.conv_output_length(input_shape[1], self.fsize, 
                                                                       padding="valid", stride=1, dilation=1)
        return (input_shape[0],) + (new_size,) + (self.fnum,)

def gammatone(band, t):
    epsilon = 1e-10  # Small number to prevent division by zero
    band = tf.maximum(band, epsilon)
    n = 4  # filter order
    sigma = ((band / 2) * tf.math.sqrt(1 / (2 ** (1 / n) - 1)))
    gammatone = (t ** (n - 1)) * tf.math.exp(-2 * math.pi * sigma * t)
    gammatone = gammatone / tf.reduce_max(gammatone)
    return gammatone
