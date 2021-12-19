#!/usr/bin/env python3

import numpy as np
from utils.face_align_torch import similarity_estimate


def _umeyama(src, dst, estimate_scale):
    """Estimate N-D similarity transformation with or without scaling.

    Parameters
    ----------
    src : (M, N) array
        Source coordinates.
    dst : (M, N) array
        Destination coordinates.
    estimate_scale : bool
        Whether to estimate scaling factor.

    Returns
    -------
    T : (N + 1, N + 1)
        The homogeneous similarity transformation matrix. The matrix contains
        NaN values only if the problem is not well-conditioned.

    References
    ----------
    .. [1] "Least-squares estimation of transformation parameters between two
            point patterns", Shinji Umeyama, PAMI 1991, :DOI:`10.1109/34.88573`

    """


    num = src.shape[0]
    # print(num)
    dim = src.shape[1]
    # print(dim)

    # Compute mean of src and dst.
    src_mean = src.mean(axis=0)
    # print(src_mean)
    dst_mean = dst.mean(axis=0)
    # print(dst_mean)

    # Subtract mean from src and dst.
    src_demean = src - src_mean
    # print(src_demean)
    dst_demean = dst - dst_mean
    # print(dst_demean)

    # Eq. (38).
    A = dst_demean.T @ src_demean / num
    # print(A)

    # Eq. (39).
    d = np.ones((dim,), dtype=np.double)
    if np.linalg.det(A) < 0:
        d[dim - 1] = -1
    # print(d)

    T = np.eye(dim + 1, dtype=np.double)
    # print(T)

    U, S, V = np.linalg.svd(A)
    # print(U)
    # print(S)
    # print(V)

    # Eq. (40) and (43).
    rank = np.linalg.matrix_rank(A)
    # print(rank)
    if rank == 0:
        return np.nan * T
    elif rank == dim - 1:
        if np.linalg.det(U) * np.linalg.det(V) > 0:
            T[:dim, :dim] = U @ V
        else:
            s = d[dim - 1]
            d[dim - 1] = -1
            T[:dim, :dim] = U @ np.diag(d) @ V
            d[dim - 1] = s
    else:
        T[:dim, :dim] = U @ np.diag(d) @ V
        # print(T)

    if estimate_scale:
        # Eq. (41) and (42).
        scale = 1.0 / src_demean.var(axis=0).sum() * (S @ d)
    else:
        scale = 1.0

    T[:dim, dim] = dst_mean - scale * (T[:dim, :dim] @ src_mean.T)
    T[:dim, :dim] *= scale

    return T


src = np.array([[34, 60], [67, 35], [65, 68], [60, 88], [95, 64]],
                   dtype=np.float32)

dst = np.array([[38.2946, 51.6963],
 [73.5318, 51.5014],
 [56.0252, 71.7366],
 [41.5493, 92.3655],
 [70.7299, 92.2041]], dtype=np.float32)

params1 = _umeyama(src, dst, True)
print(params1)
params2 = similarity_estimate(src, dst)
print(params2)
