import cv2
import os
import time
import pyspark as ps


def create_directory(directory):

    if not os.path.exists(directory):
        os.makedirs(directory)


def crop_and_resize_images(path, new_path, img_size):

    create_directory(new_path)
    direcs = [l for l in os.listdir(path) if l != '.DS_Store']

    for item in direcs:
        # Read all images as grayscale
        img = cv2.imread(path + item, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (img_size, img_size))
        cv2.imwrite(str(new_path + item), img)


if __name__ == '__main__':
    start_time = time.time()
    crop_and_resize_images(path='../data/images/', new_path='../data/resized-512/', img_size=256)
    print("Seconds: ", time.time() - start_time)
