from __future__ import print_function, division
import pandas as pd
import torch
from detecto import core, utils, visualize
import cv2
import datetime
from urllib.request import urlopen
import numpy as np
device = torch.device("cuda")
model = core.Model(['figure'])
model.load('figure_model_weights_latest4.pth',['figure'])
index = ['color', 'color_name', 'hex', 'R', 'G', 'B']
df = pd.read_csv('colors.csv', names = index, header = None)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
video_writer = cv2.VideoWriter('output.avi',fourcc, 20.0, (640,480))
url="http://192.168.43.42:80"
CAMERA_BUFFER_SIZE= 4096
stream=urlopen(url + "/stream.jpg")
bts=b''
while True:
    k = cv2.waitKey(1)
    bts += stream.read(CAMERA_BUFFER_SIZE)
    jpghead = bts.find(b'\xff\xd8')
    jpgend = bts.find(b'\xff\xd9')
    figure = 1
    if jpghead > -1 and jpgend > -1:
        jpg = bts[jpghead:jpgend + 2]
        bts = bts[jpgend + 2:]
        frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        frame = cv2.resize(frame, (640, 480))
        frame = cv2.rotate(frame, cv2.ROTATE_180)
        labels, boxes, scores = model.predict(frame)
        dt = str(datetime.datetime.now())
        cv2.putText(frame, str(dt), (40, 40), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 2, cv2.LINE_AA)
        for i in range(boxes.shape[0]):
            if scores[i] < 0.70:

                continue
            box = boxes[i]
            figure += 1
            k = int((box[1] + box[3]) / 2)
            l = int((box[0] + box[2]) / 2)
            if (k >= 480 or l >= 480):
                print('out of bound')
            else:
                cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 1)
                b, g, r = frame[k, l]
                b = int(b)
                g = int(g)
                r = int(r)
                minimum = 10000
                for i in range(len(df)):
                    d = abs(r - int(df.loc[i, "R"])) + abs(g - int(df.loc[i, "G"])) + abs(b - int(df.loc[i, "B"]))
                    if (d <= minimum):
                        minimum = d
                        cname = df.loc[i, 'color_name']
                cv2.putText(frame, cname, (box[0], box[1]), 1, 0.8, (0, 0, 255), 1, cv2.LINE_AA)
        if (figure >= 8):
            cv2.putText(frame, f'Total figure : EXCEEDED', (40, 70), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 0, 0),2)
        else:
            cv2.putText(frame, f'Total figure : {figure - 1}', (40, 70), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 0, 0),2)
        video_writer.write(frame)
        cv2.imshow('DroneF', frame)
    # If the 'q' or ESC key is pressed, break from the loop
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == 27:
        break
cv2.destroyWindow('DroneF')
video_writer.release()