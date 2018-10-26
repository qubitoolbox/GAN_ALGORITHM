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
                               optimizer=optimizer,
                               metrics=['accuracy'])
    #create the generator
    self.gen = self.build_generator()
    #The generator will take random noise as the z values and will create images
    depth = Input(shape=(self.dim,))
    images = self.gen(depth)
    #we look to only train the generator of images
    self.discriminator.trainable = False
    #Validate the input generated as images
    validation_ = self.discriminator(images)
    
    #connect the generative network with the discriminator also called adversary
    #perform training
    self.joined = Model(depth, validation_)
    self.joined.compile(loss='binary_crossentropy', optimizer)
    
  def build_generator(self):
    #initialize the type of model in tensorflow
    #with its parameters for the network
    model = Sequential()
    model.add(Dense(256, xdim = self.dim))
    model.add(optimizer)
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(512))
    model.add(optimizer)
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(1024))
    model.add(optimizer)
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(np.prod(self.pix_format), activation='tanh'))
    model.add(Reshape(self.img_shape))
    
    model.summary()
    
    
    
                               
   
