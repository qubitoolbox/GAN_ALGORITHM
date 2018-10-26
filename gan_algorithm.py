#import all libraries

from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
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
    model.add(Reshape(self.pix_format))
    
    model.summary()
    noise = Input(shape=(self.xdim,))
    images = model(noise)
    return Model(noise, images)
    
   
  def build_discriminator(self):
    #build network and its parameters
    model = Sequential()
    model.add(Flatten(self.pix_format))
    model.add(Dense(512))
    model.add(optimizer)
    model.add(Dense(256))
    model.add(optimizer))
    model.add(Dense(1, Activation='relu'))
    model.summary()
    
    images = Input(shape=self.pix_format))
    validation = model(images)
    
    return Model(images, validation)
  
  def training(self, iterations, batch= 128, num_samples=50):
    #load the data set
    (xinputrain, h_), (_h, y) = mnist.load_data()
    # normalize data in the range of -1 to 1
    xinputrain = xinput / 127.5 -1.0
    xinputrain = np.expand_dims(xinputrain, axis=3)
    
    #adversarial
    vald = np.ones((batch, 1))
    fakefoe = np.zeros((batch, 1))
    
    #iterate through back prop
    for iter in range(iterations):
    
      #select a batch of random images for training
      inrand = np.random.randint(0, xinputrain[0], batch)
      images = xinputrain[inrand]
      noise = np.random.rand(0, 1, (batch, self.pixdim))
      #generate the back of new imgs
      generated_images = self.gen.predict(noise)
      
      #train the adversarial network
      loss_func_real = self.discriminator.train_on_batch(images, vald)
      loss_func_fake = self.discriminator.train_on_batch(generated_images, fakefoe)
      loss_func = 0.5 * np.add(loss_func_real, loss_func_fake)
      
      noise = np.random.rand(0,1, (batch_size, self.pixdim))
      
      #train the generative network to validate/assimilate the samples
      loss_func_gen = self.joined.train_on_batch(noise, vald)
      #print the grandient descent before it reaches a global minima
      print ("%d [g loss: %f, acc.: %.2f%%] [d loss: %f]" % (epoch, loss_func_real[0], 10*10*loss_func_real[1], loss_func_fake))
      
      #
  def generate_images(self, iterations):
    g, r = 5,5
    noise = np.random.normal(0,1, (g * c, self.pixdim))
    generateImage = self.gen.predict(noise)
    generateImage = 0.5 * generateImage + 0.5
    
    chrt, axiss = plt.subplots(g, r)
    counter = 0
    for k in range(r):
      for j in range(g):
        axiss[k, j].imshow(generateImage[counter, :,:, 0], cmap='gray')
        axiss[k, j].axis('off')
        counter = counter + 1
    chrt.savefig("%d.png" % epoch)
    plt.close()
    
  if __name__ == '__main__':
    gan = Generative()
    gan.training(iterations=7500, batch=32, num_samples=50)
    
                               
   
