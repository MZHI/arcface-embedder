# -*- coding utf-8 -*-

import torch
from insightface.iresnet import model_urls, IBasicBlock, IResNet

weights_paths = {
    'iresnet34': ("%s/iresnet34.pth", [3, 4, 6, 3]),
    'iresnet50': ("%s/iresnet50.pth", [3, 4, 14, 3]),
    'iresnet100': ("%s/iresnet100.pth", [3, 13, 30, 3])
}


def _iresnet_local(base_path, arch, block, layers, **kwargs):
    model = IResNet(block, layers, **kwargs)
    model.load_state_dict(torch.load(weights_paths[arch][0] % base_path))
    return model


def iresnet_local(base_path, arch='iresnet100', **kwargs):
    return _iresnet_local(base_path, arch, IBasicBlock, weights_paths[arch][1], **kwargs)
