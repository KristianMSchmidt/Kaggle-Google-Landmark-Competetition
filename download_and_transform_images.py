"""
Script reads images, transforms them to size 224x224x3 and forward propagates them through
ResNet50 (minus final layer). Then it saves them all in a numpy array.

Right now the script only reads 100 images of each of the top 10 layers.

Scripts also trains final layer with 10 classes. Training the final layer only takes few seconds because
the features have been transformed.
"""

import csv, cv2, os
import numpy as np
import sys
import urllib.request
import copy

from keras.models import load_model


# Load pre-trained model (ResNet50 without top layer)

print("Loading model...")
model = load_model('ResNet50_no_top_model.h5')
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Find most common landmarks
import pandas as pd
train_data = pd.read_csv('url_data/train.csv')
from collections import Counter
c = Counter(train_data['landmark_id'])
print("Top 25 landmarks")
print("id, count")
for landmark_id, count in c.most_common(1000):
    print(landmark_id, count)

# 10 most_common_landmarks (script will only store images of these landmarks)
# id       count
# 9633     50337
# 6051     50148
# 6599     23415
# 9779     18471
# 2061     13271
# 5554     11147
# 6651      9508
# 6696      9222
# 5376      9216
# 2743      8997

#4352 8993
#13526 8667
#1553 7814
#10900 7038
#8063 6662
#8429 6426
#4987 5358
#12220 5313
#11784 5259
#2949 4919
#12718 3810
#3804 3695
#10184 3622
#7092 3543
#10045 3452

train_num = 200   # How many train images of each landmark do you want?


image_size = (224, 224)   # Original images will be resized to this size before
                          # being forward propagated through network



def transform_data(data_file):
    landmark_ids = []
    transformed_images = []


    num_saved = {'9633':0, '6051':0, '6599':0, '9779':0, '2061':0,
                 '5554':0, '6651':0, '6696':0, '5376':0, '2743':0,
                 '4352':0, '13526':0, '1553':0, '10900':0, '8063': 0,
                 '8429':0, '4987':0, '12220':0, '11784':0, '2949':0,
                '12718':0, '3804':0, '10184':0, '7092':0, '10045':0}


    csvfile = open(data_file, 'r')
    csvreader = csv.reader(csvfile)
    next(csvreader)

    for line in csvreader:
        id, url, landmark_id = line
        if landmark_id in num_saved and num_saved[landmark_id] < train_num:
            try:
                resp = urllib.request.urlopen(url, timeout = 5)
                image = np.asarray(bytearray(resp.read()), dtype="uint8")
                image = cv2.imdecode(image, cv2.IMREAD_COLOR)
                image = cv2.resize(image, image_size)
                #image = image/255  #dont uncomment this.
                image = image.reshape(1, 224, 224, 3)
                pred = model.predict(image)
                transformed_images.append(pred)
                landmark_ids.append(int(landmark_id))
                num_saved[landmark_id] += 1
                print(num_saved)
            except:
                print("Failed to download or process image:", id)

    image_array = np.stack(transformed_images)
    landmark_array = np.array(landmark_ids)

    np.savez_compressed('data2/transformed_data_top25_200_of_each', X_train = image_array, y_train = landmark_array)

#transform_data(data_file = 'url_data/train.csv')


# LOAD DATA
loaded = np.load('data2/transformed_data_top25_200_of_each.npz')
X_train = loaded['X_train']
y_train = loaded['y_train']

print(X_train.shape)
input("pause")

# Shuffle data
np.random.seed(0)
np.random.shuffle(X_train)
np.random.seed(0)
np.random.shuffle(y_train)

# Re-label data (we only want 10 categories)
d = {9633:0, 6051:1, 6599:2, 9779:3, 2061:4, 5554:5, 6651:6, 6696:7, 5376:8, 2743:9}

d = {9633:0, 6051:1, 6599:2, 9779:3, 2061:4,
     5554:5, 6651:6, 6696:7, 5376:8, 2743:9,
     4352:10, 13526:11, 1553:12, 10900:13, 8063: 14,
     8429:15, 4987:16, 12220:17, 11784:18, 2949:19,
     12718:20, 3804:21, 10184:22, 7092:23, 10045:24}

for i in range(len(y_train)):
    y_train[i] = d[y_train[i]]

# Just checking...
from collections import Counter
print(Counter(y_train))

# Transform y_train to categorical
from keras.utils.np_utils import to_categorical
y_train = to_categorical(y_train)


# Define shallow model with 10 output classes:
from keras.layers import Input, Add, Dense, Flatten, BatchNormalization, Activation, Dropout
from keras.models import Model, load_model

from keras import regularizers
#model.add(Dense(64, input_dim=64,
#                kernel_regularizer=regularizers.l2(0.01),
#                activity_regularizer=regularizers.l1(0.01)))
X_input = Input((1, 1, 1, 2048))
X = Flatten()(X_input)
#X = Dropout(0.5)(X)
#X = BatchNormalization()(X)
#X = Dense(164, activation='relu')(X)
#X = Dropout(0.5)(X)
#X = BatchNormalization()(X)
X = Dense(25, activation='softmax')(X)

#X = Dense(25, activation='softmax', kernel_regularizer=regularizers.l2(0.1))(X)

model = Model(inputs = X_input, outputs = X, name='Model')

# Compile model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
#model.summary()

print("Training...")
model.fit(X_train, y_train, epochs=15, batch_size=16, validation_split=0.10, verbose=1)
