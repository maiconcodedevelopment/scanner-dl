import keras

digit_input = keras.layers.Input(shape=(10,10,1))
modal = keras.models.Sequential()
modal.add(keras.layers.Conv2D(10,kernel_size=3,activation='relu',input_shape=digit_input,padding='same',dilation_rate=2,strides=1))

#calculating the size of the ouput

#where

#i = size of the input
#k = size of the kernel
#p = size of the zero padding
#strides

#Padding allows a convolutional layer to retain the resolution of the input into
#this layer. This is done by adding zeros around the edges of the inpu
#t image, so that the convolution kernel can overlap with the pixels on the edge of the image.

#o = ((i - k + 2p) / s) + 1 ex : 28 = ((28 - 3 + 2) / 1) + 1