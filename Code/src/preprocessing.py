import os
from PIL import Image
from pyspark import SparkContext, SparkConf
import numpy as np

APP_NAME = "ImageResizer"

def main(sc):
    size_64 = (256,256)
    i = Image.open("Amol.jpg")
    #fn, fext = os.path.splitext(f)
    #print(i.size)
    i.thumbnail(size_64)
    #print(i.size)
    #i.save('resize/{}_64'.format(fn, fext))

if __name__ == "__main__":
    conf = SparkConf().setAppName(APP_NAME)
    #   conf = conf.setMaster("spark://10.233.70.48:7077")
    sc = SparkContext()
    main(sc)
