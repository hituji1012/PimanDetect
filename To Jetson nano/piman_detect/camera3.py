import cv2
from PIL import Image, ImageFont, ImageDraw
import scipy.io
import scipy.misc
import numpy as np
import argparse
import os
import pandas as pd
import time
import cv2

import tensorflow as tf
from keras import backend as K
from keras.layers import Input, Lambda, Conv2D
from keras.models import load_model, Model
from yolo_utils import read_classes, read_anchors, generate_colors, preprocess_image, draw_boxes, scale_boxes
from keras_yolo import yolo_head, preprocess_true_boxes, yolo_loss, yolo_body,yolo_eval,tiny_yolo_body

GST_STR = 'nvarguscamerasrc \
    ! video/x-raw(memory:NVMM), width=3280, height=2464, format=(string)NV12, framerate=(fraction)30/1 \
    ! nvvidconv ! video/x-raw, width=(int)1920, height=(int)1080, format=(string)BGRx \
    ! videoconvert \
    ! appsink'
WINDOW_NAME = 'Camera Test'

def main():
    sess = K.get_session()
    image_size = 384
    image_input = Input(shape=(image_size, image_size, 3))
    dataname = 'Piman'
    class_names = [dataname]
    YOLO_ANCHORS = np.array(
    ((0.57273, 0.677385), (1.87446, 2.06253), (3.33843, 5.47434),
     (7.88282, 3.52778), (9.77052, 9.16828)))
    anchors = YOLO_ANCHORS

    yolo = tiny_yolo_body(image_input, len(anchors), len(class_names))
    yolo.load_weights('tiny_weights.h5')
 
    image_shape = (1080., 1920.) 
    yolo_outputs = yolo_head(yolo.output, anchors, len(class_names))
    boxes, scores, classes = yolo_eval(yolo_outputs, image_shape)
   
    cap = cv2.VideoCapture(GST_STR, cv2.CAP_GSTREAMER)
    count = 176
    if True:
        fourcc = cv2.VideoWriter_fourcc(*'DIVX')
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter('pimanDetect2.avi',fourcc,fps,(width,height))
    while True:
        ret, img = cap.read()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        if ret != True:
            break

        resized_image = img.resize(tuple(reversed((384, 384))), Image.BICUBIC)
        image_data = np.array(resized_image, dtype='float32')
        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.
        
        out_scores, out_boxes, out_classes = sess.run([scores, boxes, classes], feed_dict={yolo.input: image_data, K.learning_phase(): 0})
        print('Found {} boxes for {}'.format(len(out_boxes), "hoge"))
        # Generate colors for drawing bounding boxes.
        colors = generate_colors(class_names)
        # Draw bounding boxes on the image file
        draw_boxes(img, out_scores, out_boxes, out_classes, class_names, colors)
        
        cv2.imshow(WINDOW_NAME, np.asarray(img)[..., ::-1])
        writer.write(np.asarray(img)[..., ::-1])
        key = cv2.waitKey(10)
        if key == 27: # ESC
            break
    writer.release()
    cap.release()
if __name__ == "__main__":
    main()
