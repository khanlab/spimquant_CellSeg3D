from pathlib import Path
from napari_cellseg3d.dev_scripts import colab_training as c
from napari_cellseg3d.code_models.models.wnet.model import WNet
from napari_cellseg3d.config import WNetTrainingWorkerConfig, WandBConfig, WeightsInfo, PRETRAINED_WEIGHTS_DIR
import torch
import numpy as np
import os
import shutil
from napari_cellseg3d import config, utils
from cvpl_tools.dataset_reference import DatasetReference
from dataclasses import dataclass
from napari_cellseg3d.func_variants import normalize
from monai.inferers import sliding_window_inference


@dataclass
class CellSeg3DModelConfig:
    trainset: DatasetReference
    scratch_folder: str  # the folder where intermediate files for training will be written to
    result_folder: str  # result folder for output
    in_channels: int = 1
    out_channels: int = 1
    num_classes: int = 10
    dropout: float = 0.65
    IM_TRAIN_CHANNEL: int = 0  # the channel of input image to train on
    input_brightness_range: tuple[float, float] = (0., 1000.)
    number_of_epochs: int = 5
    n_cuts_weight: float = .5
    rec_loss_weight: float = .005
    rec_loss: str = 'MSE'
    device: str = 'cpu'
    model_weight_path: str = None
    post_processing_code: str = None


def preprocess_to_3d_tiles(dataset_ref: DatasetReference,
                           out_prefix, ch, clamp_max, clamp_min=0):
    """Preprocess a 4d slice (ch, z, y, x) file by selecting one of its channel and turn the volume into
    3d tiles of a given size;
    If the tile_width is None, then the slice is not tiled"""
    akd = dataset_ref.datapoint_refs

    for imid in akd:
        im = akd[imid].read_as_np(dataset_ref.im_read_setting)
        if ch is not None:
            im = im[ch]
        assert im.ndim == 3, f'Expected end results as 3-d tiles, got {im.ndim}-d tile of shape {im.shape} instead'
        im = np.clip(im, clamp_min, clamp_max)
        np.save(f'{out_prefix}/{imid}.npy', im)


def inference_on_np_batch(config: CellSeg3DModelConfig, im3d_np_batch, roi_size, model=None):
    """
    im3d_np is 5d array convertible to float32, its dimensions are (batch, in_channel, z, y, x)
    """
    if model is None:
        model = create_model(config)
    with torch.no_grad():
        model.eval()
        val_data = np.float32(im3d_np_batch)
        val_inputs = torch.from_numpy(val_data).to(config.device)
        rg = config.input_brightness_range
        if rg is None:
            for i in range(val_inputs.shape[0]):
                for j in range(val_inputs.shape[1]):
                    im_min, im_max = val_inputs.min(), val_inputs.max()
                    normalize(val_inputs[i, j], im_max=im_max, im_min=im_min, inplace=True)
        else:
            im_min, im_max = rg
            normalize(val_inputs, im_max=im_max, im_min=im_min, inplace=True)
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


def inference_on_np3d(config: CellSeg3DModelConfig, im3d_np, roi_size, model=None):
    """
    im3d_np is 3d array convertible to float32, its dimensions are (z, y, x)
    """
    if model is None:
        model = create_model(config)
    return inference_on_np_batch(config, im3d_np[None, None], roi_size, model)


def create_model(config: CellSeg3DModelConfig):
    model = WNet(
        in_channels=config.in_channels,
        out_channels=config.out_channels,
        num_classes=config.num_classes,
        dropout=config.dropout,
    )
    model.to(config.device)
    weights = torch.load(config.model_weight_path, map_location=config.device)
    model.load_state_dict(weights, strict=True)
    return model

