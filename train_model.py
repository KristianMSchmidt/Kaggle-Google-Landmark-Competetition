"""
Loads custom ResNet50-model with weights pre-trained on ImageNet.
Then retrains the final layer of this model (currently 10 classes in this layer)

Validation acc was 61% after one epoch. Model overfits, probably due to lack of
training data.

Questions:
1) How many training examples are needed to get better val accuracy?
2) How will data augmentation help overfitting problem?


With 200 of each landmark; acc: 0.9750 - val_loss: 2.5238 - val_acc: 0.6509. took 6000 seconds.
Worse results after 4 epochs:   acc: 0.9943 - val_loss: 2.9818 - val_acc: 0.6291Val acc.

"""

from keras.models import Model, load_model
import os

# Load pre-trained model
print("Loading model")
#model = load_model('model_trained_2_epochs.h5')
model = load_model('ResNet50_model3.h5')
#model = load_model('my_trained_model.h5')

# Only train last layer:
for layer in model.layers[:-1]:
    layer.trainable = False

model.summary()

#Compile model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


# Fit model
from tensorflow.python.keras.applications.resnet50 import preprocess_input
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator

image_size = 224
# The ImageDataGenerator was previously generated with
# data_generator = ImageDataGenerator(preprocessing_function=preprocess_input)
# recent changes in keras require that we use the following instead:
#data_generator = ImageDataGenerator()



data_generator = ImageDataGenerator(featurewise_center=False,
                                    samplewise_center=False,
                                    featurewise_std_normalization=False,
                                    samplewise_std_normalization=False,
                                    zca_whitening=False,
                                    zca_epsilon=1e-06,
                                    rotation_range=0.0,
                                    width_shift_range=0.0,
                                    height_shift_range=0.0,
                                    shear_range=0.0,
                                    zoom_range=0.0,
                                    channel_shift_range=0.0,
                                    fill_mode='nearest',
                                    cval=0.0,
                                    horizontal_flip=False,
                                    vertical_flip=False,
                                    rescale=None,
                                    preprocessing_function=None,
                                    data_format=None)

path = "C://Users//kimar//Google Drev//Python programmer//Kaggle//Landmark_recognition2//data"

train_generator = data_generator.flow_from_directory(
        os.path.join(path, 'train'),
        target_size=(image_size, image_size),
        batch_size=15,
        class_mode='categorical')

#print(train_generator.classes)
#print("")

validation_generator = data_generator.flow_from_directory(
       os.path.join(path, 'val'),
       target_size=(image_size, image_size),
       batch_size = 10,
       class_mode='categorical')

#print(validation_generator.class_indices)
#print(validation_generator.classes)

# steps_per_epoch should be (number of training images total / batch_size)
# validation_steps should be (number of validation images total / batch_size)

model.fit_generator(
       train_generator,
       steps_per_epoch=10   ,
       validation_data=validation_generator,
       validation_steps=1,
       epochs = 3,
       callbacks = 
       )

#Save trained model
print("Saving model")
model.save('my_trained_model.h5')
