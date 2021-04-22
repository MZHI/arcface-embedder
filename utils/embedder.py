# -*- coding utf-8 -*-

import torch
import insightface
from torchvision import transforms
from utils.utils_local_weights import  iresnet100local, iresnet34local, iresnet50local


class Embedder():
    def __init__(self, is_local, weights_base_path=None):
        self.__device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        if is_local:
            # load embedder from local models
            self.__embedder = iresnet100local(weights_base_path)
        else:
            # load embedder from remote urls
            self.__embedder = insightface.iresnet100(pretrained=True)

        self.__embedder.to(self.__device)
        self.__embedder.eval()

        # check if model on GPU
        # print(next(embedder.parameters()).is_cuda)

        mean = [0.5] * 3
        std = [0.5 * 256 / 255] * 3
        self.__preprocess = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(112),
            transforms.CenterCrop(112),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

    def get_features(self, face):
        tensor = self.__preprocess(face)
        tensor = tensor.to(self.__device)

        with torch.no_grad():
            features = self.__embedder(tensor.unsqueeze(0))[0]
        return features


