from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from dataUtil import *



if __name__=="__main__":
    face_dataset=FaceLandmarksDataset(csv_file='./faces/face_landmarks.csv',root_dir='./faces/')
    print(len(pd.read_csv('./faces/face_landmarks.csv'))) 
    print(len(face_dataset))  #69
    scale = Rescale(256)
    crop = RandomCrop(128)
    composed = transforms.Compose([Rescale(256),
                               RandomCrop(224)])

    # Apply each of the above transforms on sample.
    fig = plt.figure()
    sample = face_dataset[65]
    for i, tsfrm in enumerate([scale, crop, composed]):
        transformed_sample = tsfrm(sample)

        ax = plt.subplot(1, 3, i + 1)
        plt.tight_layout()
        ax.set_title(type(tsfrm).__name__)
        show_landmarks(**transformed_sample)

    plt.show()



