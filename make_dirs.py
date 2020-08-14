import os, csv

def make_folders(data_file):
    csvfile = open(data_file, 'r')
    csvreader = csv.reader(csvfile)
    next(csvreader)
    landmarks = set()
    for line in csvreader:
        try:
            _, _, landmark_id = line
            landmarks.add(landmark_id)
        except:
            print("last line")
    for landmark_id in landmarks:
        out_dir_val = os.path.join('all_data','val', landmark_id)
        out_dir_train = os.path.join('all_data','train', landmark_id)

        if not os.path.exists(out_dir_val):
            os.makedirs(out_dir_val)
        if not os.path.exists(out_dir_train):
            os.makedirs(out_dir_train)
    print("done")

make_folders('url_data/train.csv')
