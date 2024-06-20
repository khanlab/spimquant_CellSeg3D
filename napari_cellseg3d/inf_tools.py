from napari_cellseg3d.code_models.models.wnet.model import WNet
import torch
import numpy as np
from napari_cellseg3d.utils import remap_image
from monai.inferers import sliding_window_inference
from monai.transforms import AsDiscrete
from .func_variants import normalize


def inference_on_np(model, im3d_np, roi_size, rg, device):
    """
    im3d_np can be any 3d array convertible to float32
    """
    with torch.no_grad():
        model.eval()
        for _k, val_data_file in enumerate(image_files):
            val_data = np.float32(im3d_np[None, None, :])
            val_inputs = torch.from_numpy(val_data).to(device)
            if rg is None:
                im_min, im_max = val_inputs.min(), val_inputs.max()
            else:
                im_min, im_max = rg
            normalize(val_inputs[0, 0], im_max=im_max, im_min=im_min, inplace=True)
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
            yield val_outputs


def inference_on(model, image_files, roi_size, rg, device):
    """
    :param model:
    :param image_files: one file or many files
    :return:
    """
    for _k, val_data_file in enumerate(image_files):
        val_outputs = inference_on_np(model, np.load(val_data_file), roi_size, rg, device)
        yield val_outputs


