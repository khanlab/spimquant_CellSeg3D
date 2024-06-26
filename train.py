from napari_cellseg3d.dev_scripts import colab_training as c
from napari_cellseg3d.config import WNetTrainingWorkerConfig, WandBConfig, WeightsInfo, PRETRAINED_WEIGHTS_DIR
from napari_cellseg3d import utils
from spimquant import CellSeg3DModelConfig
from cvpl_tools.dataset_reference import DatasetReference
import numpy as np
import os, shutil
from pathlib import Path
import argparse
import cvpl_tools.strenc as strenc
import json


def main():
    config = load_config_from_argparse_for_training()
    train_model(config)


def load_config_from_argparse_for_training() -> CellSeg3DModelConfig:
    parser = argparse.ArgumentParser(description='Example with dictionary')
    parser.add_argument('--im_channel', type=int, default=0,
                        help='The channel of input image to train CellSeg3D on')
    parser.add_argument('--trainset', type=str, default=None,
                        help='path to the dataset reference file; this is the training set to use')
    parser.add_argument('--min_brightness', type=float, default=0.,
                        help='The minimum brightness threshold to clip the input image into')
    parser.add_argument('--max_brightness', type=float, default=1000.,
                        help='The maximum brightness threshold to clip the input image into')
    parser.add_argument('--n_cuts_weight', type=float, default=.5,
                        help='Weighting of one of the two losses used by CellSeg3D: n cuts loss')
    parser.add_argument('--rec_loss_weight', type=float, default=.005,
                        help='Weighting of one of the two losses used by CellSeg3D: reconstruction loss')
    parser.add_argument('--nepoch', type=int, default=20,
                        help='The number of epochs to train for')
    parser.add_argument('--num_classes', type=int, default=.005,
                        help='The number of output class in the autoencoder representation of CellSeg3D')
    parser.add_argument('--scratch_folder', type=str, default=None,
                        help='Path to the scratch folder that can be used for storing intermediate input files.')
    parser.add_argument('--result_folder', type=str, default=None,
                        help='Path to the result folder that can be used for storing result files (model weights etc.)')
    parser.add_argument('--device', type=str, default='cpu', help='device to use for training')

    args = parser.parse_args()
    with open(args.trainset, 'r') as infile:
        ref: DatasetReference = json.load(infile, object_hook=strenc.get_decoder_hook())
    config = CellSeg3DModelConfig(
        trainset=ref,
        scratch_folder=args.scratch_folder,
        result_folder=args.result_folder,
        im_channel=args.im_channel,
        input_brightness_range=(args.min_brightness, args.max_brightness),
        n_cuts_weight=args.n_cuts_weight,
        rec_loss_weight=args.rec_loss_weight,
        nepochs=args.nepochs,
        num_classes=args.num_classes,
        device=args.device,
    )

    return config


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
                                 config.im_channel,
                                 clamp_max, clamp_min)

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
        max_epochs=config.nepochs,
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


if __name__ == '__main__':
    main()
