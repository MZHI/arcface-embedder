#!/usr/bin/env python3
# -*- coding utf-8 -*-

import argparse
import cv2
import sys
from utils.detector import Detector
from utils.embedder import Embedder


def main(args):

    image_path = args.image_path
    is_local_weights = args.is_local_weights
    weights_base_path = args.weights_base_path
    show_face = args.show_face

    embedder_path = "pytorch-insightface/resource"

    # detect faces from image
    image = cv2.imread(image_path)

    det = Detector()
    boxes, landmarks = det.detect(image)

    if boxes.shape[0] < 1:
        print("Faces not found")
        sys.exit(0)

    # get first face detection
    box = boxes[0, :].cpu().numpy()

    # crop face
    x_tl, y_tl, x_br, y_br = box[0], box[1], box[2], box[3]
    face = image[y_tl:y_br, x_tl:x_br, :]

    if show_face:
        # show face crop
        cv2.imshow("Detected face.", face)
        cv2.waitKey(0)

    embedder = Embedder(is_local_weights, weights_base_path)

    features = embedder.get_features(face)

    print("Features calculation finished. ")
    print(f"Features shape: {features.shape}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser('''Face detector and embedder''')

    parser.add_argument('--image-path', type=str, default="./images/office5.jpg",
                        help="Path to image to be processed")
    parser.add_argument('--is-local-weights', type=int, default=0,
                        help="Whether to use local weights or from remote server")
    parser.add_argument('--weights-base-path', type=str, default="pytorch-insightface/resource",
                        help="Root path to insightface weights, converted to PyTorch format. "
                             "Actual only if --is-local-weights == 1")
    parser.add_argument('--show-face', type=int, default=0,
                        help="Whether to show cropped face or not")
    args = parser.parse_args()
    main(args)
