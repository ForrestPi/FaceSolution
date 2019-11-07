import pyLibfacedetection_cnn
import cv2

capture = cv2.VideoCapture()
capture.open(0)#capture.open('./demo16/test.avi')

faceDetect = pyLibfacedetection_cnn.facedetect

while True:
    ret, frame = capture.read()
    if not ret:
        print('Finish!')
        break

    Face = faceDetect(frame)
    for face in Face:
        cv2.rectangle(frame, (face.rect[0], face.rect[1]), (face.rect[2], face.rect[3]), (0, 255, 255), 2)
    cv2.imshow('tracking', frame)
    cv2.waitKey(33)