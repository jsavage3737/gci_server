# imports
import os
import keras
import shutil
import numpy as np
import pandas as pd
from keras import losses
from keras.applications.xception import preprocess_input
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator

# building xception model
xception = keras.applications.xception.Xception(include_top=True, 
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

# xception
xception.compile(optimizer='sgd', loss='binary_crossentropy', metrics =['accuracy'])

xception.fit_generator(generator=train_generator,
                    steps_per_epoch=64,
                    validation_data=valid_generator,
                    validation_steps=24,
                    epochs=10
)


xception.save('/home/paperspace/GCI/scripts/model_1/xception_model_3')


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
xception_predicts = xception_model_1.preidct_generator(valid_generator,verbose=1)
xception_metrics = xception_model_1.evaluate_generator(valid_generator,verbose=1)

# saving predictions and metrics for analysis
np.savetxt("xception_validation_predicts1.csv",xception_predicts,delimiter=",")
np.savetxt("xception_validation_metrics1.csv",xception_metrics,delimiter=",")

