import cv2
import matplotlib.pyplot as plt
import numpy as np

img_file = './girl.jpg'
img = cv2.imread(img_file)
img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
plt.figure(figsize=(10, 10))
plt.imshow(img)
plt.show()

import sys
sys.path.insert(0,"D://research//FaceDetector")
import mtcnn
from mtcnn.utils import draw

#加载mtcnn模型
# First we create pnet, rnet, onet, and load weights from caffe model.
pnet, rnet, onet = mtcnn.get_net_caffe('../output/converted')

# Then we create a detector
detector = mtcnn.FaceDetector(pnet, rnet, onet, device='cpu')

img = cv2.imread(img_file)
boxes, landmarks = detector.detect(img, minsize=24)
face = draw.crop(img, boxes=boxes, landmarks=landmarks)[0]
face = cv2.cvtColor(face, cv2.COLOR_RGB2BGR)
plt.figure(figsize=(5, 5))
plt.imshow(face)
plt.show()


# Define the correct points.
REFERENCE_FACIAL_POINTS = np.array([
    [30.29459953,  51.69630051],
    [65.53179932,  51.50139999],
    [48.02519989,  71.73660278],
    [33.54930115,  92.3655014],
    [62.72990036,  92.20410156]
], np.float32)

# Lets create a empty image|
empty_img = np.zeros((112,96,3), np.uint8) 
draw.draw_landmarks(empty_img, REFERENCE_FACIAL_POINTS.astype(int))

plt.figure(figsize=(5, 5))
plt.imshow(empty_img)
plt.show()

img_copy = img.copy()
landmark = landmarks[0]

img_copy[:112, :96, :] = empty_img
img_copy = cv2.cvtColor(img_copy, cv2.COLOR_RGB2BGR)
draw.draw_landmarks(img_copy, landmark)

plt.figure(figsize=(15, 15))
plt.imshow(img_copy)
plt.show()

trans_matrix = cv2.getAffineTransform(landmark[:3].cpu().numpy().astype(np.float32), REFERENCE_FACIAL_POINTS[:3])


aligned_face = cv2.warpAffine(img.copy(), trans_matrix, (112, 112))
aligned_face = cv2.cvtColor(aligned_face, cv2.COLOR_RGB2BGR)

plt.figure(figsize=(5, 5))
plt.imshow(aligned_face)
plt.show()


from numpy.linalg import inv, norm, lstsq
from numpy.linalg import matrix_rank as rank

def findNonreflectiveSimilarity(uv, xy, K=2):

    M = xy.shape[0]
    x = xy[:, 0].reshape((-1, 1))  # use reshape to keep a column vector
    y = xy[:, 1].reshape((-1, 1))  # use reshape to keep a column vector

    tmp1 = np.hstack((x, y, np.ones((M, 1)), np.zeros((M, 1))))
    tmp2 = np.hstack((y, -x, np.zeros((M, 1)), np.ones((M, 1))))
    X = np.vstack((tmp1, tmp2))

    u = uv[:, 0].reshape((-1, 1))  # use reshape to keep a column vector
    v = uv[:, 1].reshape((-1, 1))  # use reshape to keep a column vector
    U = np.vstack((u, v))

    # We know that X * r = U
    if rank(X) >= 2 * K:
        r, _, _, _ = lstsq(X, U)
        r = np.squeeze(r)
    else:
        raise Exception('cp2tform:twoUniquePointsReq')

    sc = r[0]
    ss = r[1]
    tx = r[2]
    ty = r[3]

    Tinv = np.array([
        [sc, -ss, 0],
        [ss,  sc, 0],
        [tx,  ty, 1]
    ])


    T = inv(Tinv)

    T[:, 2] = np.array([0, 0, 1])

    T = T[:, 0:2].T

    return T

similar_trans_matrix = findNonreflectiveSimilarity(landmark.cpu().numpy().astype(np.float32), REFERENCE_FACIAL_POINTS)

aligned_face = cv2.warpAffine(img.copy(), similar_trans_matrix, (112, 112))
aligned_face = cv2.cvtColor(aligned_face, cv2.COLOR_RGB2BGR)

plt.figure(figsize=(5, 5))
plt.imshow(aligned_face)
plt.show()