import keras
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
import numpy as np
#Builds a ConvNet with 2 Conv layers (8 and then 16 filters) 1 Max pooling, 1 dense 32 neuron layer and then 2 outputs
def ConvNet(shape):
  model_input = Input(shape)
  layers = Conv2D(8, (3,3), activation='relu', padding = 'same')(model_input)
  layers = Conv2D(16, (3,3), activation='relu', padding = 'same')(layers)
  layers = MaxPooling2D((2, 2), strides=(2, 2))(layers)
  layers = Flatten()(layers)
  layers = Dense(32, activation='relu')(layers)
  layers = Dense(2, activation='softmax')(layers)
  model = Model(model_input, layers)
  model.compile(optimizer='adadelta', loss='categorical_crossentropy', metrics=['accuracy'])
  return model
#Builds a data generator for the supplied dir
def getDataGenerator(_dir,  img_width, img_height, batch_size):
  datagen = ImageDataGenerator(rescale=1./255)
  generator = datagen.flow_from_directory(_dir,  target_size=(img_width, img_height), 
                                          batch_size=batch_size, class_mode='categorical') 
  return generator

def main():
  batch_size = 5
  img_width = 150
  img_height = 150
  #Get a data generator for the train data
  train_generator = getDataGenerator("data/train", img_width, img_height, batch_size)
  test_generator = getDataGenerator("data/test", img_width, img_height, batch_size)
  #Initalise a model
  model = ConvNet((img_width,img_height,3))
  #Fit the data to the model
  model.fit_generator(train_generator, steps_per_epoch= (40 // batch_size), epochs=10 )
  print("Accuracy on Test Data: %f" %(model.evaluate_generator(test_generator)[1]))

if __name__ == '__main__':
  main()