#!/usr/bin/env python3
# -*- coding utf-8 -*-

import argparse
import cv2
import sys
import torch
import numpy as np
from utils.detector import Detector
from utils.embedder import Embedder
from utils.face_align_numpy import align_face_np
from utils.face_align_torch import align_face_torch_batch


def main(args):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    image_path = args.image_path
    is_local_weights = args.is_local_weights
    weights_base_path = args.weights_base_path
    show_face = args.show_face
    align_torch = args.align_torch
    arch = args.arch

    # detect faces from image
    image = cv2.imread(image_path)

    det = Detector()
    boxes, landmarks = det.detect(image)
    print(f"Found faces: {boxes.shape[0]}")

    if boxes.shape[0] < 1:
        print("Faces not found")
        sys.exit(0)

    if align_torch:
        faces_aligned, _ = align_face_torch_batch(image, landmarks, boxes, device)
    else:
        faces_aligned, _ = align_face_np(image, landmarks, boxes)

    if show_face:
        # show detected face
        idx = 0
        x_tl, y_tl, x_br, y_br = boxes[idx, 0], boxes[idx, 1], boxes[idx, 2], boxes[idx, 3]
        face = image[y_tl:y_br, x_tl:x_br, :]
        cv2.imshow("Detected face", face)

        if align_torch:
            face_aln = faces_aligned.cpu().numpy()[idx, :, :, :].squeeze().copy()
            print(face_aln.min())
            print(face_aln.max())
            face_aln = (face_aln * 255).astype(np.uint8)
            face_aln = face_aln.transpose(1, 2, 0)
        else:
            face_aln = faces_aligned[idx].squeeze().copy().astype(np.uint8)

        cv2.imshow("Aligned face", face_aln)
        cv2.waitKey(0)

    # create embedder and get features
    embedder = Embedder(is_local_weights, arch, weights_base_path)

    features = embedder.get_features(faces_aligned)
    # print(features[idx, :])

    print("Features calculation finished. ")
    print(f"Features shape: {features.shape}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser('''Face detector and embedder''')

    parser.add_argument('--image-path', type=str, default="./images/office5.jpg",
                        help="Path to image to be processed. Default: ./images/office5.jpg")
    parser.add_argument('--is-local-weights', type=int, default=0,
                        help="Whether to use local weights or from remote server. Default: 0")
    parser.add_argument('--weights-base-path', type=str, default="pytorch-insightface/resource",
                        help="Root path to insightface weights, converted to PyTorch format. "
                             "Actual only if --is-local-weights == 1. Default: pytorch-insigntface/resource")
    parser.add_argument('--show-face', type=int, default=0,
                        help="Whether to show cropped face or not. Default: 0")
    parser.add_argument('--align-torch', type=int, default=1,
                        help="Whether to use torch or numpy realization of alignment. Default: 1")
    parser.add_argument('--arch', type=str, default="iresnet100",
                        help="Architecture of embedder. [iresnet34/iresnet50/iresnet100]. Default: iresnet100")
    args = parser.parse_args()
    main(args)
