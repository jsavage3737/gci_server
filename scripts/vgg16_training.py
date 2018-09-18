# imports
import os
import keras
import shutil
import numpy as np
import pandas as pd
from keras import losses
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator

# building vgg 16 model
vgg16 = keras.applications.vgg16.VGG16(include_top=True, 
                                       weights=None, 
                                       input_tensor=None, 
                                       input_shape=(224,224,1), 
                                       pooling=None, 
                                       classes=2)


# image loading + preproccessing
datagen = ImageDataGenerator() ###### EDIT THIS LATER IF OVERFITTING BECOMES AN ISSUE

print("Loading training data")
train_generator = datagen.flow_from_directory(
    directory= '\home\paperspace\GCI\data/train_jpgs'.replace('\\','/'),
    target_size=(224, 224),
    color_mode="grayscale",
    batch_size=42,
    class_mode="categorical",
    shuffle=True,
)

print("Loading validation data")
valid_generator = datagen.flow_from_directory(
    directory= r'\home\paperspace\GCI\data/validation_jpgs'.replace('\\','/'),
    target_size=(224, 224),
    color_mode="grayscale",
    batch_size=12,
    class_mode="categorical",
    shuffle=True,
)

# vgg16
vgg16.compile(optimizer='sgd', loss='binary_crossentropy', metrics =['accuracy'])

vgg16.fit_generator(generator=train_generator,
                    steps_per_epoch=64,
                    validation_data=valid_generator,
                    validation_steps=24,
                    epochs=10
)


vgg16.save('/home/paperspace/GCI/scripts/model_1/vgg16_model2')


################################# MODEL TESTING ###########

# loads data unshuffled for categorical separation 
valid_generator = datagen.flow_from_directory(
    directory= r'\home\paperspace\GCI\data/validation_jpgs'.replace('\\','/'),
    target_size=(224, 224),
    color_mode="grayscale",
    batch_size=12,
    class_mode="categorical",
    shuffle=False,
)

# generating predictions and metrics off validation set
vgg16_predicts = vgg_model_1.preidct_generator(valid_generator,verbose=1)
vgg16_metrics = vgg_model_1.evaluate_generator(valid_generator,verbose=1)

# saving predictions and metrics for analysis
np.savetxt("vgg16_validation_predicts1.csv",vgg16_predicts,delimiter=",")
np.savetxt("vgg16_validation_metrics1.csv",vgg16_metrics,delimiter=",")

