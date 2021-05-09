import cv2
import numpy as np


########### PARAMETERS ###########
cap = cv2.VideoCapture('test.avi')
whT = 320
Threshold = 0.5
nmsThreshold = 0.3
########### PARAMETERS ###########

path = 'coco.names'
classNames = []

with open(path, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')
#print(classNames)

model_cfg = 'yolov3.cfg'
model_weights = 'yolov3.weights'

net = cv2.dnn.readNetFromDarknet(model_cfg, model_weights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

def findObjects(outputs, img):
    hT , wT , cT = img.shape
    bbox = []
    classIds = []
    confs = []

    total_humans = 0

    for output in outputs:
          for det in output:
              scores = det[5:]
              classId = np.argmax(scores)
              confidence = scores[classId]

              if confidence > Threshold:
                  w,h = int(det[2]*wT) , int(det[3]*hT)
                  x,y = int((det[0]*wT) - w/2) , int((det[1]*hT) - h/2)
                  bbox.append([x,y,w,h])
                  classIds.append(classId)
                  confs.append(float(confidence))

                  if(classNames[classId]=='person'):
                      total_humans+=1

    #print(len(bbox))
    #print(total_humans)
    indices = cv2.dnn.NMSBoxes(bbox, confs, Threshold, nmsThreshold)

    for i in indices:
        i = i[0]
        box = bbox[i]
        x,y,w,h = box[0] , box[1] , box[2], box[3]
        cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,255), 2)
        #cv2.putText(img, f'{classNames[classIds[i]].upper()} {confs[i]*100}%',
                    #(x,y-10),cv2.FONT_HERSHEY_COMPLEX_SMALL,0.6, (255,0,255) ,2)


while True:
    suc, img = cap.read()

    blob = cv2.dnn.blobFromImage(img, 1/255, (whT,whT), [0,0,0], 1, crop=False)
    net.setInput(blob)

    layerNames = net.getLayerNames()

    outputNames = [layerNames[i[0]-1] for i in net.getUnconnectedOutLayers()]
    outputs = net.forward(outputNames)

    findObjects(outputs, img)

    cv2.imshow('Image',img)
    cv2.waitKey(1)
