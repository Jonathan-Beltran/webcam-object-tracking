#use open cv to open up webcam and read frames and use yolo v8 to detect objects in frame
#draw box around detected objects and display final image
import cv2
from ultralytics import YOLO
import pandas as pd

model = YOLO("yolov8l.pt")  # load a pretrained model (recommended for training)

# Opens the inbuilt camera of laptop to capture video.
cap = cv2.VideoCapture(0)
i = 0

while (cap.isOpened()):
    ret, frame = cap.read()

    # This condition prevents from infinite looping
    # incase video ends.
    if ret == False:
        break

    detections = model(frame)[0]

    id_to_name = detections.names

    for detection in detections:
        boxes = detection[0].boxes.numpy()  # Boxes object for bbox outputs
        for box in boxes:  # there could be more than one detection

            name = id_to_name[box.cls[0]]
            print("class", box.cls)
            print("name", name)
            print("xyxy", box.xyxy)
            print("conf", box.conf)

            conf = box.conf

            box = box.xyxy[0]

            point1 = [box[0], box[1]]
            point2 = [box[2], box[3]]

            point1 = [int(x) for x in point1]
            point2 = [int(x) for x in point2]

            if conf > 0.75:

                frame = cv2.rectangle(frame, point1, point2, (10,0,250), 2)
                frame = cv2.putText(frame, name, org=point1, fontFace=0, fontScale=5, color=(0,255,0), thickness=2)


    cv2.imshow("Camera", frame)
    cv2.waitKey(1)

    i += 1

cap.release()
cv2.destroyAllWindows()