#!/usr/bin/env python3

import cv2
import sys
import mtcnn
import insightface
import torch
from imageio import imread
from torchvision import transforms

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# First we create pnet, rnet, onet, and load weights from caffe model.
pnet, rnet, onet = mtcnn.get_net_caffe('output/converted')

# Then we create a detector
detector = mtcnn.FaceDetector(pnet, rnet, onet, device='cuda:0')

# Then we can detect faces from image
img = './images/office5.jpg'
image = cv2.imread(img)
boxes, landmarks = detector.detect(image)

# if boxes.shape[0] < 1:
#     sys.exit(0)

# Then we draw bounding boxes and landmarks on image
image = cv2.imread(img)
image = mtcnn.utils.draw.draw_boxes2(image, boxes)
image = mtcnn.utils.draw.batch_draw_landmarks(image, landmarks)

# Show the result
cv2.imshow("Detected image.", image)
cv2.waitKey(0)
