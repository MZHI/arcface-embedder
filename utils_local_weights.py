import torch
from insightface.iresnet import model_urls, IBasicBlock, IResNet

model_paths = {
    'iresnet34': "%s/iresnet34.pth",
    'iresnet50': "%s/iresnet50.pth",
    'iresnet100': "%s/iresnet100.pth"
}


def _iresnet_local(base_path, arch, block, layers, **kwargs):
    model = IResNet(block, layers, **kwargs)
    model.load_state_dict(torch.load(model_paths[arch] % base_path))
    return model


def iresnet34local(base_path, progress=True, **kwargs):
    return _iresnet_local(base_path, 'iresnet34', IBasicBlock, [3, 4, 6, 3],
                    **kwargs)


def iresnet50local(base_path, **kwargs):
    return _iresnet_local(base_path, 'iresnet50', IBasicBlock, [3, 4, 14, 3],
                    **kwargs)


def iresnet100local(base_path, **kwargs):
    return _iresnet_local(base_path, 'iresnet100', IBasicBlock, [3, 13, 30, 3],
                    **kwargs)