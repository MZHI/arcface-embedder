# -*- coding utf-8 -*-

import mtcnn

class Detector:
    def __init__(self):
        # First we create pnet, rnet, onet, and load weights from caffe model.
        pnet, rnet, onet = mtcnn.get_net_caffe('output/converted')

        # Then we create a detector
        self.__detector = mtcnn.FaceDetector(pnet, rnet, onet, device='cuda:0')

    def detect(self, image):
        boxes, landmarks = self.__detector.detect(image)
        return boxes, landmarks

