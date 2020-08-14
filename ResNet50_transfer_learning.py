"""
Transfer learning using resnet50 without its final layer.Using sequential model.
Adding new top layer and only training this layer.

Problem: Model cannot be saved and reloaded.
"""
from tensorflow.python.keras.applications import ResNet50
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Flatten, GlobalAveragePooling2D


num_classes = 10
resnet_weights_path = 'C://Users//kimar//Google Drev//Python programmer//Kaggle//DeepLearningTrack//Dog_Breed//data//resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'

my_new_model = Sequential()
my_new_model.add(ResNet50(include_top=False, pooling='avg', weights=resnet_weights_path))
my_new_model.add(Dense(num_classes, activation='softmax'))

# Say not to train first layer (ResNet) model. It is already trained
my_new_model.layers[0].trainable = False


#my_new_model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
my_new_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

my_new_model.summary()

input("")

# Fit model
from tensorflow.python.keras.applications.resnet50 import preprocess_input
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator

image_size = 224
# The ImageDataGenerator was previously generated with
# data_generator = ImageDataGenerator(preprocessing_function=preprocess_input)
# recent changes in keras require that we use the following instead:
data_generator = ImageDataGenerator()

train_generator = data_generator.flow_from_directory(
        'data/train',
        target_size=(image_size, image_size),
        batch_size=16,
        class_mode='categorical')

print(train_generator.classes)
print("")
validation_generator = data_generator.flow_from_directory(
        'data/val',
        target_size=(image_size, image_size),
        class_mode='categorical')

#print(validation_generator.class_indices)
#print(validation_generator.classes)

input("")
my_new_model.fit_generator(
        train_generator,
        steps_per_epoch=62,

        epochs = 4,
        validation_data=validation_generator,
        validation_steps=4)

#Save trained model
print("Saving model")
my_new_model.save('model.h5')


from keras.models import load_model
try:
    model = load_model('model.h5')
except:
    print("load error")


############### Try to predict some images with model #########33
import numpy as np
from tensorflow.python.keras.applications.resnet50 import preprocess_input
from tensorflow.python.keras.preprocessing.image import load_img, img_to_array


#Choose Images to Work With
from os.path import join

image_dir = 'data/val/'
img_paths = [join(image_dir, filename) for filename in
                          ['2061/6b6e78da3e9d2771.jpg',
                           '2743/2f79bc0106cc36b5.jpg']]
image_size = 224

def read_and_prep_images(img_paths, img_height=image_size, img_width=image_size):
    """
    Function to Read and Prep Images for Modeling
    """

    #imgs = [load_img(img_path, target_size= img_height, img_width)) for img_path in img_paths]
    imgs = [load_img(img_path, target_size=None) for img_path in img_paths]

    img_array = np.array([img_to_array(img) for img in imgs])
    return preprocess_input(img_array)


a_few_images = read_and_prep_images(img_paths, img_height=224, img_width = image_size)

#Make predictions
preds = my_new_model.predict(a_few_images)
print(100*preds)
