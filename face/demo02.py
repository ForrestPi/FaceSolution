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
    fig=plt.figure()
    for i in range(len(face_dataset)):
        sample = face_dataset[i]
        print(i,sample['image'].shape,sample['landmarks'].shape)
        ax=plt.subplot(1,4,i+1)
        plt.tight_layout()
        ax.set_title('Sample #{}'.format(i))
        ax.axis('off')
        show_landmarks(**sample)

        if i == 3:
            plt.show()
            break