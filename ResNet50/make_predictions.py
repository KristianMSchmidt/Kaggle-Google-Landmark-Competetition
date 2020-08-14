
from keras.models import load_model
model = load_model('model_trained_1_epoch.h5')

############### Try to predict some images with model #########33
import numpy as np
from tensorflow.python.keras.applications.resnet50 import preprocess_input
from tensorflow.python.keras.preprocessing.image import load_img, img_to_array


#Choose Images to Work With
from os.path import join
image_dir = 'C://Users//kimar//Google Drev//Python programmer//Kaggle//Landmark_recognition2//data//val'

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

preds = model.predict(a_few_images)


print("Prediction probabilities in percent:", 100*preds)
