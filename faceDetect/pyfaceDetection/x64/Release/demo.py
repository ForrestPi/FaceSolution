import face_tracking_demo as demo
import cv2

capture = cv2.VideoCapture()
capture.open(0)#capture.open('./demo16/test.avi')

tracker = demo.FaceTracker()
tracker.trackerInit(model_path='./models/', min_face=40)

while True:
    ret, frame = capture.read()
    if not ret:
        print('Finish!')
        break

    rect = tracker.trackerUpdate(frame)

    cv2.rectangle(frame, (rect[0], rect[1]), (rect[2], rect[3]), (0, 255, 255), 2)
    cv2.imshow('tracking', frame)
    cv2.waitKey(33)