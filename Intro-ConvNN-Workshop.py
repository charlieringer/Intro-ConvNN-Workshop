import keras
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
import numpy as np

def ConvNet(shape):
  model_input = Input(shape)
  layers = Conv2D(8, (3,3), activation='relu', padding = 'same')(model_input)
  layers = Conv2D(16, (3,3), activation='relu', padding = 'same')(layers)
  layers = MaxPooling2D((2, 2), strides=(2, 2))(layers)
  layers = Flatten()(layers)
  layers = Dense(32, activation='relu')(layers)
  layers = Dense(2, activation='softmax')(layers)
  model = Model(model_input, layers)
  model.compile(optimizer='adadelta', loss='binary_crossentropy', metrics=['accuracy'])
  return model

def getDataGenerator(dir,  img_width, img_height, batch_size):
  datagen = ImageDataGenerator(rescale=1./255)
  generator = datagen.flow_from_directory(
        dir,  
        target_size=(img_width, img_height), 
        batch_size=batch_size,
        class_mode='categorical') 
  return generator

def main():
  batch_size = 5
  img_width = 150
  img_height = 150

  train_generator = getDataGenerator("data/train", img_width, img_height, batch_size)

  model = ConvNet((img_width,img_height,3))
  model.fit_generator(train_generator, steps_per_epoch= 40 // batch_size, epochs=10 )

main()