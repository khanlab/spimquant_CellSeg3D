from pathlib import Path
from napari_cellseg3d.dev_scripts import colab_training as c
from napari_cellseg3d.code_models.models.wnet.model import WNet
from napari_cellseg3d.config import WNetTrainingWorkerConfig, WandBConfig, WeightsInfo, PRETRAINED_WEIGHTS_DIR
import torch
import numpy as np
import os
import shutil
from napari_cellseg3d import config, utils
from monai.transforms import LoadImaged, Compose
from monai.data import DataLoader, Dataset
import logging
import monai
from spimquant.cvpl_tools.dataset_reference import DatasetReference
from dataclasses import dataclass
from .func_variants import normalize


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


def train_model(config: CellSeg3DModelConfig):
    PROCESSED_SLICE_FOLDER = f'{config.scratch_folder}/cellseg3d_train_inputs'
    if os.path.exists(PROCESSED_SLICE_FOLDER):
        shutil.rmtree(PROCESSED_SLICE_FOLDER)
    os.mkdir(PROCESSED_SLICE_FOLDER)
    if os.path.exists(config.result_folder):
        shutil.rmtree(config.result_folder)
    os.mkdir(config.result_folder)

    clamp_min, clamp_max = config.input_brightness_range
    preprocess_to_3d_tiles(config.trainset, PROCESSED_SLICE_FOLDER,
                                 config.IM_TRAIN_CHANNEL,
                                 clamp_max, clamp_min)

    do_validation = False

    batch_size = 4
    learning_rate = 2e-5
    weight_decay = 0.01
    validation_frequency = 2
    intensity_sigma = 1.0
    spatial_sigma = 4.0
    ncuts_radius = 2

    train_data_folder = Path(PROCESSED_SLICE_FOLDER)

    def create_dataset(folder):
        images_filepaths = utils.get_all_matching_files(folder, pattern={'.npy', })
        data_dict = [{"image": str(image_name)} for image_name in images_filepaths]
        return data_dict

    WANDB_INSTALLED = False  # we don't use wandb

    global train_config, wandb_config
    train_config = WNetTrainingWorkerConfig(
        device=config.device,
        max_epochs=config.number_of_epochs,
        learning_rate=learning_rate,
        validation_interval=validation_frequency,
        batch_size=batch_size,
        num_workers=2,
        weights_info=WeightsInfo(),
        results_path_folder=config.result_folder,
        train_data_dict=create_dataset(train_data_folder),
        eval_volume_dict=None,
        # advanced
        in_channels=config.in_channels,
        out_channels=config.out_channels,
        num_classes=config.num_classes,
        dropout=config.dropout,
        weight_decay=weight_decay,
        intensity_sigma=intensity_sigma,
        spatial_sigma=spatial_sigma,
        radius=ncuts_radius,
        reconstruction_loss=config.rec_loss,
        n_cuts_weight=config.n_cuts_weight,
        rec_loss_weight=config.rec_loss_weight,
        input_brightness_range=config.input_brightness_range,
    )
    wandb_config = WandBConfig(
        mode="disabled" if not WANDB_INSTALLED else "online",
        save_model_artifact=False,
    )

    worker = c.get_colab_worker(worker_config=train_config, wandb_config=wandb_config)
    for epoch_loss in worker.train():
        continue
    config.model_weight_path = f'{config.result_folder}/wnet.pth'
    return config


def train_channel(channel: int, trainset: DatasetReference, scratch_folder: str, result_folder: str, device: str):
    if channel == 0:
        MAX_BRIGHTNESS = 1000.
        rec_loss_weight = .005
    else:
        MAX_BRIGHTNESS = 500.
        rec_loss_weight = .2

    config = CellSeg3DModelConfig(
        trainset=trainset,
        scratch_folder=scratch_folder,
        result_folder=result_folder,
        IM_TRAIN_CHANNEL=channel,
        input_brightness_range=(0., MAX_BRIGHTNESS),
        number_of_epochs=10,
        n_cuts_weight=.5,
        rec_loss_weight=.005,
        rec_loss='MSE',
        device=device,
    )
    train_model(config)
    return config


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


def inference_on(config: CellSeg3DModelConfig, im3d_np_batch, roi_size, model=None):
    """
    :param model:
    :param image_files: one file or many files
    :return:
    """
    if model is None:
        model = create_model(config)
    for _k, val_data_file in enumerate(image_files):
        val_outputs = inference_on_np3d(model, np.load(val_data_file), roi_size, model)
        yield val_outputs


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

