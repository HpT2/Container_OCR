import cv2
import numpy as np


def build_model(classes_path='./yolo/yolov4.txt', weights_path='./yolo/yolov4.weights', config_path='./yolo/yolov4.cfg'):
    # read class names from text file
    with open(classes_path, 'r') as f:
        classes = [line.strip() for line in f.readlines()]
    # read pre-trained model and config file to create the network
    net = cv2.dnn.readNet(weights_path, config_path)
    layer_names = net.getLayerNames() 
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()] 
    return net, classes, output_layers


#detect code region
def detect(net, frame, output_layers):
    # initialization
    class_ids = []
    confidences = []
    boxes = []
    CONFIDENCE_THRESHOLD = 0.5
    NMS_THRESHOLD = 0.4
    scale = 1/255
    size = (320, 320)
    # create input blob to prepare image for the network
    blob = cv2.dnn.blobFromImage(frame, scalefactor=scale, size=size, mean=(0, 0, 0), swapRB=True, crop=False)
    net.setInput(blob)
    im_h, im_w = frame.shape[0:2]
    # run inference through the network and gather predictions from output layers
    outs = net.forward(output_layers)
    if outs:
    # for each detection from each output layer, get the confidence, class id, bounding box params
    # and ignore weak detections (confidence < 0.5)
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.1:
                    # convert yolo coords to opencv coords
                    center_x = int(detection[0] * im_w)
                    center_y = int(detection[1] * im_h)
                    w = int(detection[2] * im_w)
                    h = int(detection[3] * im_h)
                    x = center_x - w / 2
                    y = center_y - h / 2
                    class_ids.append(class_id)
                    confidences.append(float(confidence))
                    boxes.append([x, y, w, h])
        # apply non-max suppression
        indices = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
         # clean up
        clean_class_ids = []
        clean_confidences = []
        clean_boxes = []
        for i in indices:
            j = i
            clean_class_ids.append(class_ids[j])
            clean_boxes.append(boxes[j])
            clean_confidences.append(confidences[j])
        return clean_class_ids, clean_boxes, clean_confidences

#crop bounding box
def crop(frame, boxes):
    x,y,w,h = boxes[0]
    cropped = frame[round(y):round(y+h)+1,round(x):round(x+w)+1]
    return cropped

#detect and crop code region
def detect_code(img):
     net,_, output_layers = build_model()
     _, boxes,_ = detect(net, img, output_layers)
     code_reg = crop(img, boxes)
     return code_reg
