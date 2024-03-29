# -*- coding utf-8 -*-

import torch
import numpy as np
import insightface
from torchvision import transforms
from utils.local_weights import iresnet_local


class Embedder():
    def __init__(self, is_local, arch="iresnet100", weights_base_path=None):
        self.__device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        if is_local:
            # load weights from locally
            self.__embedder = iresnet_local(weights_base_path, arch)
        else:
            # load weights from remote urls
            if arch == "iresnet34":
                self.__embedder = insightface.iresnet34(pretrained=True)
            elif arch == "iresnet50":
                self.__embedder = insightface.iresnet50(pretrained=True)
            elif arch == "iresnet100":
                self.__embedder = insightface.iresnet100(pretrained=True)
            else:
                raise ValueError("Invalid arch type for embedder")

        self.__embedder.to(self.__device)
        self.__embedder.eval()

        # check if model on GPU
        # print(next(embedder.parameters()).is_cuda)

        mean = [0.5] * 3
        std = [0.5 * 256 / 255] * 3
        self.__preprocess_basic = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(112),
            transforms.CenterCrop(112),
            transforms.ToTensor(),
        ])
        self.__normalize = transforms.Normalize(mean, std)

    def get_features(self, faces):
        if not isinstance(faces, torch.Tensor):
            face_tensors = [self.__preprocess_basic(face) for face in faces]
            faces = torch.stack(face_tensors, dim=0)
        faces = self.__normalize(faces)
        faces = faces.to(self.__device)

        with torch.no_grad():
            if len(faces.shape) == 3:
                features = self.__embedder(faces.unsqueeze(0))
            else:
                features = self.__embedder(faces)
        return features


