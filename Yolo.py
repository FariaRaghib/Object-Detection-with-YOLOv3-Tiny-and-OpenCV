import cv2
import numpy as np

# first working on the frames
cap = cv2.VideoCapture(0)  # 0 bcz we have only one camera connected to it
whT = 320
classesFile = 'coco.names'
classNames = []
confThreshold=0.5
nmsThreshold=0.3
with open(classesFile, 'rt') as f:  # open classesFiles and read the text file as f
    classNames = f.read().rstrip('\n').split('\n')
print(classNames)
print(len(classNames))

modelConfiguration = 'yolov3_tiny.cfg'
modelWeights = 'yolov3-tiny.weights'

net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_Target_CPU)


# NET ACCEPTS SPECIFIC TYPE OF FORMAT WHICH IS BLOB
def findObjects(outputs, img):
    hT, wT, cT = img.shape
    bbox = []
    classIds = []
    confs = []

    for output in outputs:
        for det in output:

            scores = det[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > confThreshold:
                w, h = int(det[2]*wT),int(det[3]*hT)
                x,y = int((det[0] * wT)-w/2), int(det[1]*hT-h/2)
                bbox.append([x,y,w,h])
                classIds.append(classId)
                confs.append(float(confidence))
        print(len(bbox))
        indices = cv2.dnn.NMSBoxes(bbox,confs,confThreshold,nmsThreshold)

        for i in indices:
            i=i[0]
            box=bbox[i]
            x,y,w,h=box[0],box[1],box[2],box[3]
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,255),2)
            cv2.putText(img,f'{classNames[classIds[i]].upper()}{int(confs[i]*100)}%',
                        (x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,0,255),2)

while True:
    success, img = cap.read()  # data successfully retrive or not
    # so we have to convert this image to blob
    blob = cv2.dnn.blobFromImage(img, 1 / 255, (whT, whT), [0, 0, 0], 1, crop=False)
    net.setInput(blob)
    layerNames = net.getLayerNames()
    # print(layerNames)

    outputNames = [layerNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    # print(net.getUnconnectedOutLayers())
    # print(outputNames)
    outputs = net.forward(outputNames)
    print(outputs[0].shape)
    print(outputs[1].shape)
    print(outputs[2].shape)

    findObjects(outputs,img)
    cv2.imshow('Image', img)
    cv2.waitKey(1)  # DELAY FOR 1 MILLISEC
