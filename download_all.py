import csv, cv2, os
import numpy as np
import sys
import urllib.request
import copy


image_size = (224, 224)   # Original images will be resized to this size before
                          #being saved

def save_image(train_val, id, url, landmark_id):

    filename = os.path.join('all_data', train_val, landmark_id, '{}.jpg'.format(id))
    if os.path.exists(filename):
        print("File already exists")
        return True

    try:
        resp = urllib.request.urlopen(url, timeout = 1)
        image = np.asarray(bytearray(resp.read()), dtype="uint8")
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        image = cv2.resize(image, image_size)
        #image = image/255
        cv2.imwrite(filename, image)
        return True
    except:
        print("Failed to download or process image:", id)


def read_csv(data_file):
    #Jeg reserverer de foerste 10000 linjer til validation
    val_num = 10000

    csvfile = open(data_file, 'r')
    csvreader = csv.reader(csvfile)
    next(csvreader)

    count = 0
    for line in csvreader:
        print(line)
        input("pause")
        count +=1
        #if count < 64000:
        #    continue


        if count < 10000 and count%10 == 0:
            print("Image number:", count)
        if count >= 10000 and count%100 == 0:
            print("Image number:", count)

        try:
            id, url, landmark_id = line
        except:
            print("problem with this line in csv file", line)

        if count < val_num:
            save_image("val", id, url, landmark_id)
        else:
            save_image("train", id, url, landmark_id)

read_csv('url_data/train.csv')
