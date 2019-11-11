from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from dataUtil import *



if __name__=="__main__":
    transformed_dataset = FaceLandmarksDataset(csv_file='faces/face_landmarks.csv',
                                                root_dir='faces/',
                                                transform=transforms.Compose([
                                                    Rescale(256),
                                                    RandomCrop(224),
                                                    ToTensor()
                                                ]))
    dataloader = DataLoader(transformed_dataset, batch_size=4,
                        shuffle=True, num_workers=4)

    for i_batch, sample_batched in enumerate(dataloader):
        print(i_batch, sample_batched['image'].size(),
            sample_batched['landmarks'].size())

        # observe 4th batch and stop.
        if i_batch == 3:
            plt.figure()
            show_landmarks_batch(sample_batched)
            plt.axis('off')
            plt.ioff()
            plt.show()
            break