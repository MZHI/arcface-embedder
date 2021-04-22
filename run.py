#!/usr/bin/env python3
# -*- coding utf-8 -*-

import argparse
import cv2
import sys
import mtcnn
import insightface
import torch
from imageio import imread
from torchvision import transforms
from utils_local_weights import iresnet34local, iresnet50local, iresnet100local


def main(args):

    image_path = args.image_path
    is_local_weights = args.is_local_weights
    weights_base_path = args.weights_base_path
    show_face = args.show_face

    embedder_path = "pytorch-insightface/resource"

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # First we create pnet, rnet, onet, and load weights from caffe model.
    pnet, rnet, onet = mtcnn.get_net_caffe('output/converted')

    # Then we create a detector
    detector = mtcnn.FaceDetector(pnet, rnet, onet, device='cuda:0')

    # detect faces from image
    image = cv2.imread(image_path)
    boxes, landmarks = detector.detect(image)

    if boxes.shape[0] < 1:
        sys.exit(0)

    # Next: get first face detection
    box = boxes[0, :].cpu().numpy()

    # crop face and move to tensor
    x_tl, y_tl, x_br, y_br = box[0], box[1], box[2], box[3]
    face = image[y_tl:y_br, x_tl:x_br, :]

    if is_local_weights:
        # load embedder from local models
        embedder = iresnet100local(weights_base_path)
    else:
        # load embedder from remote urls
        embedder = insightface.iresnet100(pretrained=True)

    embedder.to(device)
    embedder.eval()

    # check if model on GPU
    # print(next(embedder.parameters()).is_cuda)

    mean = [0.5] * 3
    std = [0.5 * 256 / 255] * 3
    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(112),
        transforms.CenterCrop(112),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    tensor = preprocess(face)
    tensor = tensor.to(device)

    with torch.no_grad():
        features = embedder(tensor.unsqueeze(0))[0]

    print("Features calculation finished. ")
    print(f"Features shape: {features.shape}")

    if show_face:
        # show face crop
        cv2.imshow("Detected face.", face)
        cv2.waitKey(0)


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
