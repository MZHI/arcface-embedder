#!/usr/bin/env python3

import cv2
import sys
import mtcnn
import insightface
import torch
from imageio import imread
from torchvision import transforms
from utils_local_weights import iresnet34local, iresnet50local, iresnet100local

embedder_path = "pytorch-insightface/resource"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# First we create pnet, rnet, onet, and load weights from caffe model.
pnet, rnet, onet = mtcnn.get_net_caffe('output/converted')

# Then we create a detector
detector = mtcnn.FaceDetector(pnet, rnet, onet, device='cuda:0')

# Then we can detect faces from image
img = './images/office5.jpg'
image = cv2.imread(img)
boxes, landmarks = detector.detect(image)

if boxes.shape[0] < 1:
    sys.exit(0)

# Next: get first face detection
box = boxes[0, :].cpu().numpy()

# crop face and move to tensor
x_tl, y_tl, x_br, y_br = box[0], box[1], box[2], box[3]
face = image[y_tl:y_br, x_tl:x_br, :]

# load embedder from remote urls
# embedder = insightface.iresnet100(pretrained=True)
# load embedder from local models
embedder = iresnet100local(embedder_path)

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


# # Then we draw bounding boxes and landmarks on image
# image = cv2.imread(img)
# image = mtcnn.utils.draw.draw_boxes2(image, boxes)
# image = mtcnn.utils.draw.batch_draw_landmarks(image, landmarks)
#
# # Show the result
# cv2.imshow("Detected image.", image)
# cv2.waitKey(0)
