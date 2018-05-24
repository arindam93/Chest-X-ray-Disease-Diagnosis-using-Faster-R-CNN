import time

import cv2
import numpy as np
import pandas as pd


def convert_images_to_arrays(file_path, df):

    lst_images = [l for l in df['Image_Index']]

    return np.array([np.array(cv2.imread(file_path + img, cv2.IMREAD_GRAYSCALE)) for img in lst_images])


def save_to_array(arr_name, arr_object):

    return np.save(arr_name, arr_object)

if __name__ == '__main__':

    start_time = time.time()
    labels = pd.read_csv("../data/sample_labels.csv")
    print("Writing Train Array")
    X_train = convert_images_to_arrays('../data/resized-512/', labels)
    print(X_train.shape)
    print("Saving Train Array")
    save_to_array('../data/X_sample.npy', X_train)
    print("Seconds: ", round(time.time() - start_time), 2)