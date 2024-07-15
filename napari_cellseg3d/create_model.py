from napari_cellseg3d.code_models.models.wnet.model import WNet
import torch


def create_model(config: dict):
    model = WNet(
        in_channels=config['in_channels'],
        out_channels=config['out_channels'],
        num_classes=config['num_classes'],
        dropout=config['dropout'],
    )
    model.to(config['device'])
    weights = torch.load(config['model_weight_path'], map_location=config['device'])
    model.load_state_dict(weights, strict=True)
    return model