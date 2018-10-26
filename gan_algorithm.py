#import all libraries

from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatte, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyRelu
from keras import optimizers
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
import matplotlib.pyplot as plt
import numpy as np

class Generative():
  def __init__(self):
    self.pix_row = 28
    self.pix_cols = 25
    self.gray = 1
    self.pix_format = (self.pix_row, self.pix_cols, self.gray)
    self.dim = 75
    #set the learning learning rate and decay value
    #learning rate is set to 1.0 due to adadelta serving
    #as an 'automated' learning rate finder
    optimizer = Adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.003)
    
    #create the discriminator
    self.discriminator = self.build_discriminator()
    self.discriminator.compile(loss='mean_squared_error',
                               optimizer=
                               
   
