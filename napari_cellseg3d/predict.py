from typing import Sequence

import torch
from napari_cellseg3d.create_model import create_model
from napari_cellseg3d.func_variants import normalize
from monai.inferers import sliding_window_inference
import numpy as np


def inference_on_torch_batch(config: dict,
                             val_inputs: torch.Tensor,
                             roi_size: Sequence[int],
                             model=None,
                             var_filter=(100., .9995)) -> torch.Tensor:
    """
    inference on a batch of 3d images (represented as a 5D tensor (B, C, Z, Y, X) where C is channel)
    Args
        config - The model config specifying model settings, and path to model file if model is not provided
        val_inputs - Input tensor
        roi_size - inference window size
        model - The CellSeg3D model object to use as predictor
        var_filter - a tuple (brightness, portion) image patches with portion of value less than brightness
            greather than portion are treated as empty spaces and predicted as 0s w/o calling CellSeg3D predictor
    Returns
        result probability density tensor (5D) with axis order (B, C, Z, Y, X) where C is predicted class
    """
    if model is None:
        model = create_model(config)
    with torch.no_grad():
        model.eval()
        rg = config['input_brightness_range']
        if rg is None:
            for i in range(val_inputs.shape[0]):
                for j in range(val_inputs.shape[1]):
                    im_min, im_max = val_inputs.min(), val_inputs.max()
                    val_inputs[i, j] = normalize(val_inputs[i, j], im_max=im_max, im_min=im_min, new_max=1.,
                                                 inplace=False)
        else:
            im_min, im_max = rg
            val_inputs = normalize(val_inputs, im_max=im_max, im_min=im_min, new_max=1, inplace=False)

        def predictor(im) -> torch.Tensor:
            nonlocal var_filter
            if (im < var_filter[0]).sum() / float(im.size) > var_filter[1]:  # skip predictions on empty spaces
                pred = torch.zeros((val_inputs.shape[0], config['num_classes']) + im.shape[2:],
                                   device=config['device'], dtype=torch.float32)
            else:
                pred = model.forward_encoder(im)
            return pred

        val_outputs = sliding_window_inference(
            val_inputs,
            roi_size=roi_size,
            sw_batch_size=1,
            predictor=predictor,
            overlap=0.1,
            mode="gaussian",
            sigma_scale=0.01,
            progress=True,
        )
        return val_outputs


def inference_on_np_batch(config: dict, im3d_np_batch, roi_size, model=None) -> torch.Tensor:
    """
    im3d_np is 5d array convertible to float32, its dimensions are (batch, in_channel, z, y, x)
    """
    val_data = np.float32(im3d_np_batch)
    val_inputs = torch.from_numpy(val_data).to(config['device'])
    return inference_on_torch_batch(config, val_inputs, roi_size, model)


def inference_on_np3d(config: dict, im3d_np, roi_size, model=None) -> torch.Tensor:
    """
    im3d_np is 3d array convertible to float32, its dimensions are (z, y, x)
    """
    assert im3d_np.ndim == 3, f'Expected 3d image, got shape {im3d_np.shape}'
    if model is None:
        model = create_model(config)
    return inference_on_np_batch(config, im3d_np[None, None], roi_size, model)
