import torch
from create_model import create_model
from napari_cellseg3d.func_variants import normalize
from monai.inferers import sliding_window_inference
import numpy as np


def inference_on_np_batch(config: dict, im3d_np_batch, roi_size, model=None):
    """
    im3d_np is 5d array convertible to float32, its dimensions are (batch, in_channel, z, y, x)
    """
    if model is None:
        model = create_model(config)
    with torch.no_grad():
        model.eval()
        val_data = np.float32(im3d_np_batch)
        val_inputs = torch.from_numpy(val_data).to(config['device'])
        rg = config['input_brightness_range']
        if rg is None:
            for i in range(val_inputs.shape[0]):
                for j in range(val_inputs.shape[1]):
                    im_min, im_max = val_inputs.min(), val_inputs.max()
                    val_inputs[i, j] = normalize(val_inputs[i, j], im_max=im_max, im_min=im_min, new_max=1., inplace=False)
        else:
            im_min, im_max = rg
            val_inputs = normalize(val_inputs, im_max=im_max, im_min=im_min, new_max=1, inplace=False)
        val_outputs = sliding_window_inference(
            val_inputs,
            roi_size=roi_size,
            sw_batch_size=1,
            predictor=model.forward_encoder,
            overlap=0.1,
            mode="gaussian",
            sigma_scale=0.01,
            progress=True,
        )
        # val_decoder_outputs = sliding_window_inference(
        #     val_outputs,
        #     roi_size=roi_size,
        #     sw_batch_size=1,
        #     predictor=model.forward_decoder,
        #     overlap=0.1,
        #     mode="gaussian",
        #     sigma_scale=0.01,
        #     progress=True,
        # )
        return val_outputs


def inference_on_np3d(config: dict, im3d_np, roi_size, model=None):
    """
    im3d_np is 3d array convertible to float32, its dimensions are (z, y, x)
    """
    assert im3d_np.ndim == 3, f'Expected 3d image, got shape {im3d_np.shape}'
    if model is None:
        model = create_model(config)
    return inference_on_np_batch(config, im3d_np[None, None], roi_size, model)
