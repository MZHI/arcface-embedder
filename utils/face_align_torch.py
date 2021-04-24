# -*- coding utf-8 -*-

import numpy as np
import torch
from torchgeometry import warp_affine
from torchvision import transforms

src1 = torch.Tensor([[51.642, 50.115], [57.617, 49.990], [35.740, 69.007],
                 [51.157, 89.050], [57.025, 89.702]]).float()
#<--left
src2 = torch.Tensor([[45.031, 50.118], [65.568, 50.872], [39.677, 68.111],
                 [45.177, 86.190], [64.246, 86.758]]).float()

#---frontal
src3 = torch.Tensor([[39.730, 51.138], [72.270, 51.138], [56.000, 68.493],
                 [42.463, 87.010], [69.537, 87.010]]).float()

#-->right
src4 = torch.Tensor([[46.845, 50.872], [67.382, 50.118], [72.737, 68.111],
                 [48.167, 86.758], [67.236, 86.190]]).float()

#-->right profile
src5 = torch.Tensor([[54.796, 49.990], [60.771, 50.115], [76.673, 69.007],
                 [55.388, 89.702], [61.257, 89.050]]).float()

src = torch.stack((src1, src2, src3, src4, src5), dim=0)


# Estimate N-D similarity transformation without scaling
def similarity_estimate(src, dst, estimate_scale=True):
    if not isinstance(src, torch.Tensor):
        src = torch.from_numpy(src).float()
    if not isinstance(dst, torch.Tensor):
        dst = torch.from_numpy(dst).float()

    src = src.cpu()
    dst = dst.cpu()

    num, dim = src.shape

    # compute mean of src and dst
    src_mean = torch.mean(src, dim=0)
    dst_mean = torch.mean(dst, dim=0)

    # Subtract mean from src and dst
    src_demean = src - src_mean
    dst_demean = dst - dst_mean

    A = torch.matmul(dst_demean.t(), src_demean) / num

    d = torch.ones(dim).float()
    if torch.linalg.det(A) < 0:
        d[dim - 1] = -1

    T = torch.eye(dim+1).float()

    U, S, V = torch.svd(A)

    rank = torch.matrix_rank(A)
    if rank == 0:
        return None
    elif rank == dim - 1:
        if torch.linalg.det(U) * torch.linalg.det(V) > 0.:
            T[:dim, :dim] = torch.matmul(U, V)
        else:
            s = d[dim - 1]
            d[dim - 1] = -1
            T[:dim, :dim] = torch.matmul(torch.matmul(U, torch.diag(d).float()), V)
            d[dim - 1] = s
    else:
        T[:dim, :dim] = torch.matmul(torch.matmul(U, torch.diag(d).float()), V)

    if estimate_scale:
        scale = 1.0 / torch.var(src_demean, dim=0, unbiased=False).sum() * torch.matmul(S, d)
    else:
        scale = 1.0

    T[:dim, dim] = dst_mean - scale * (torch.matmul(T[:dim, :dim], src_mean.t()))
    T[:dim, :dim] *= scale

    return T


def estimate_norm_torch(lmks, image_size=112):
    l_min_M = []
    l_min_idxs = []
    for j in range(lmks.shape[0]):
        lmk = lmks[j, :, :]
        print(f"current lmk shape: {lmk.shape}")
        lmk_tran = torch.cat((lmk, torch.ones(5, 1)), dim=1)
        min_M = []
        min_index = []
        min_error = float('inf')
        for i in np.arange(src.shape[0]):
            params = similarity_estimate(lmk, src[i])
            M = params[:2, :]
            results = torch.matmul(M, lmk_tran.t())
            results = results.t()
            error = torch.sum(torch.sqrt(torch.sum(torch.square(results - src[i]), dim=1)))
            if error < min_error:
                min_error = error
                min_M = M
                min_index = i
        l_min_M.append(min_M)
        l_min_idxs.append(min_index)
    M = torch.stack(l_min_M, dim=0)
    return M, l_min_idxs


def norm_crop_torch(faces_tensor, lmks_tensor, device, image_size=112):
    assert len(faces_tensor.shape) == 4
    assert len(lmks_tensor.shape) == 3
    M, pose_indexes = estimate_norm_torch(lmks_tensor)
    M = M.to(device)
    faces_tensor = faces_tensor.to(device)
    warped = warp_affine(faces_tensor, M, (image_size, image_size))
    return warped, pose_indexes


def align_face_torch_batch(img_orig, landmarks, bboxes, device, image_size=112):
    """
    Face alignment using [N] batches
    :param img_orig: original image, numpy array
    :param landmarks: tensor [N*5*2]
    :param bboxes: bounding boxes of faces, [N*4]
    :param device: name of device
    :param image_size: size of image, to which each crop wil be resized
    :return:
    """
    lmks = landmarks.cpu()
    bboxes = bboxes.cpu()
    batch_size = lmks.shape[0]

    # vector of horizontal and vertical scale coefficients for [N] bboxes
    k_h_tensor = float(image_size) / (bboxes[:, 2] - bboxes[:, 0])
    k_v_tensor = float(image_size) / (bboxes[:, 3] - bboxes[:, 1])

    lmks = lmks - torch.cat([bboxes[:, :2]]*5, dim=1).view(-1, 5, 2)

    # scale landmarks and move to device
    coefs_tensor = torch.cat([torch.stack([k_h_tensor, k_v_tensor], dim=1)] * 5, dim=1).view(-1, 5, 2)
    lmks = lmks * coefs_tensor

    # create transformations
    resize_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((112, 112)),
        transforms.ToTensor()
    ])

    faces_crops = []
    for i in range(batch_size):
        # get crop of face, resize to [image_size] and move to Tensor
        faces_crops.append(img_orig[bboxes[i, 1]:bboxes[i, 3], bboxes[i, 0]:bboxes[i, 2], :])
        faces_crops[i] = resize_transforms(faces_crops[i])

    # create tensor of size [N*3*image_size*image_size]
    faces_tensor = torch.stack(faces_crops, dim=0)
    faces_tensor, pose_idx = norm_crop_torch(faces_tensor, lmks, device)
    return faces_tensor, pose_idx
