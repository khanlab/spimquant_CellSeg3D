import torch
import numpy as np
from typing import Union


def normalize(
    image: Union[np.ndarray, torch.Tensor],
    im_max,
    im_min,
    new_max=100,
    new_min=0,
    batch_size=262144,
    inplace=False,
):
    """Modified from CellSeg3D/napari_cellseg3d/utils.py for low mem inference; in place normalize
    Normalizes a numpy array or Tensor using the max and min value."""
    with torch.no_grad():
        if isinstance(image, np.ndarray):
            assert np.issubdtype(image.dtype, np.inexact)  # test if image is one of floating point types
            # numpy array
            if inplace:
                assert image.flags['C_CONTIGUOUS'], 'ERROR: Numpy array must be C contiguous to be reshaped to vector'
                im_view = np.ascontiguousarray(image)
            else:
                image = np.copy(image)
                im_view = image.reshape(-1)
        else:
            # torch tensor
            if inplace:
                im_view = image.view(-1)
            else:
                image = torch.clone(image)
                im_view = image.reshape(-1)
        for i in range(0, len(im_view), batch_size):
            iend = i + batch_size
            im_view[i:iend] = (im_view[i:iend] - im_min) / (im_max - im_min)
            im_view[i:iend] = im_view[i:iend] * (new_max - new_min) + new_min
        if inplace:
            return image
        else:
            return im_view.reshape(image.shape)

