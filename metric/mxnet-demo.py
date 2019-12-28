#https://discussion.datafountain.cn/questions/1904/answers/22795
import os
import cv2
import numpy as np
import time
import scipy.io as sio
from collections import OrderedDict
from tqdm import tqdm_notebook as tqdm
import insightface
def load_image(img_path, filp=False):
    image = cv2.imread(img_path, 1)
    image = image[-96:,:,:]
    image = cv2.resize(image,(112,112))
    if image is None:
        return None
    if filp:
        image = cv2.flip(image,1,dst=None)
    return image
model = insightface.model_zoo.get_model('arcface_r100_v1')
model.prepare(ctx_id = 0)
def get_featurs(model, test_list):
    pbar = tqdm(total=len(test_list))
    for idx, img_path in enumerate(test_list):
        pbar.update(1)
        img = load_image(img_path)
        if idx==0:
            feature = model.get_embedding(img)
            features = feature
        else:
            feature = model.get_embedding(img)
            features = np.concatenate((features, feature), axis=0)
    return features
def get_feature_dict(test_list, features):
    fe_dict = {}
    for i, each in enumerate(test_list):
        fe_dict[each] = features[i]
    return fe_dict
def cosin_metric(x1, x2):
    return np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))
data_dir = '../input/testing/'   # testset dir
name_list = [name for name in os.listdir(data_dir)]
img_paths = [data_dir+name for name in os.listdir(data_dir)]
print('Images number:', len(img_paths))
s = time.time()
features = get_featurs(model, img_paths)
t = time.time() - s
print(features.shape)
print('total time is {}, average time is {}'.format(t, t / len(img_paths)))
fe_dict = get_feature_dict(name_list, features)
print('Output number:', len(fe_dict))
sio.savemat('face_embedding_test.mat', fe_dict)
######## cal_submission.py #########
face_features = sio.loadmat('face_embedding_test.mat')
print('Loaded mat')
sample_sub = open('../input/submission_template.csv', 'r')  # sample submission file dir
sub = open('submission_new.csv', 'w') 
print('Loaded CSV')
lines = sample_sub.readlines()
pbar = tqdm(total=len(lines))
for line in lines:
    pair = line.split(',')[0]
    sub.write(pair+',')
    a,b = pair.split(':')
    score = '%.2f'%cosin_metric(face_features[a][0], face_features[b][0])
    sub.write(score+'\n')
    pbar.update(1)
sample_sub.close()
sub.close()