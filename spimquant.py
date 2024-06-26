from napari_cellseg3d.code_models.models.wnet.model import WNet
import torch
from cvpl_tools.dataset_reference import DatasetReference
from dataclasses import dataclass
import argparse


@dataclass
class CellSeg3DModelConfig:
    trainset: DatasetReference
    scratch_folder: str  # the folder where intermediate files for training will be written to
    result_folder: str  # result folder for output
    in_channels: int = 1
    out_channels: int = 1
    num_classes: int = 10
    dropout: float = 0.65
    im_channel: int = 0  # the channel of input image to train on
    input_brightness_range: tuple[float, float] = (0., 1000.)
    nepochs: int = 5
    n_cuts_weight: float = .5
    rec_loss_weight: float = .005
    rec_loss: str = 'MSE'
    device: str = 'cpu'
    model_weight_path: str = None
    post_processing_code: str = None


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

