import os
import pandas as pd


def get_lst_images(file_path):

    return [i for i in os.listdir(file_path) if i != '.DS_Store']


if __name__ == '__main__':
    data = pd.read_csv("../data/Data_Entry_2017.csv")
    resized_im = os.listdir('../data/resized-256/')
    resized_im = pd.DataFrame({'Image Index': resized_im})
    resized_im = pd.merge(resized_im, data, how='left', on='Image Index')

    resized_im.columns = ['Image_Index', 'Finding_Labels', 'Follow_Up_#', 'Patient_ID',
                      'Patient_Age', 'Patient_Gender', 'View_Position',
                      'Original_Image_Width', 'Original_Image_Height',
                      'Original_Image_Pixel_Spacing_X',
                      'Original_Image_Pixel_Spacing_Y', 'Unnamed']

    resized_im['Finding_Labels'] = resized_im['Finding_Labels'].apply(lambda x: x.split('|')[0])
    resized_im.drop(['Original_Image_Pixel_Spacing_X', 'Original_Image_Pixel_Spacing_Y', 'Unnamed'], axis=1, inplace=True)
    resized_im.drop(['Original_Image_Width', 'Original_Image_Height'], axis=1, inplace=True)
    resized_im.to_csv('../data/resized_im_labels.csv', index=False, header=True)
