"""
Scripts to download images in size 224x224x3 to 'val' and 'train' directories,
sorted by landmark_id.


My future plan is this:
1) Script should be such that an already downloaded image is skipped.
2) Retrain ResNet50-classifier. Will be interesting to see training time and
   accuracy!
3) How many images of each class is neaded? What if I add more classes - say 20?
   Or all classes with at least 100 images in each? How will data augmentation
    help compared to having many original images of each kind?
"""

import csv, cv2, os
import numpy as np
import sys
import urllib.request
import copy

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

val_num = 55      # If val_num = 12, it means that the program will try to save
                  # the first 12 occurances of each landmark as a validation
                  # image. This number should be a bit higher than de actual
                  # number of validation images that you want as some url's may fail

train_num = 300   # How many train images of each landmark do you want?
                  # No train image will be among the first val_num of each landmark,
                  # so validation images
                  # train images will not be mixed

image_size = (224, 224)   # Original images will be resized to this size before
                          #being saved



def save_image(train_val, id, url, landmark_id):
    filename = os.path.join('data', train_val, landmark_id, '{}.jpg'.format(id))
    if os.path.exists(filename):
        print("File already exists")
        return True
    out_dir = os.path.join('data',train_val, landmark_id)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    try:
        resp = urllib.request.urlopen(url, timeout = 5)
        image = np.asarray(bytearray(resp.read()), dtype="uint8")
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        image = cv2.resize(image, image_size)
        image = image/255 #Carefull: This is probably WRONG!!!
        #print(image.shape)
        #cv2.imshow("Image", image)
        #cv2.waitKey(0)
        cv2.imwrite(filename, image)
        return True
    except:
        print("Failed to download or process image:", id)

def read_csv(data_file):

    counter = {'9633':0, '6051':0, '6599':0, '9779':0, '2061':0,
               '5554':0, '6651':0, '6696':0, '5376':0, '2743':0}

    val_saved = copy.deepcopy(counter)
    train_saved = copy.deepcopy(counter)

    csvfile = open(data_file, 'r')
    csvreader = csv.reader(csvfile)
    next(csvreader)

    for line in csvreader:
        id, url, landmark_id = line
        if landmark_id in counter:
            counter[landmark_id] += 1

            if val_saved[landmark_id] < val_num:
                if save_image("val", id, url, landmark_id):
                    val_saved[landmark_id] += 1
                    print("Val saved:", val_saved.items())

            elif train_saved[landmark_id] < train_num:
                if save_image("train", id, url, landmark_id):
                    train_saved[landmark_id] += 1
                    print("Trn saved:", train_saved.items())

    print("")
    print("Done!")
    print("Total validation images saved", val_saved.items())
    print("Total train images saved", train_saved.items())
    print("")
    print("Total number of occurences of most common landmarks:", counter.items())

read_csv('url_data/train.csv')
