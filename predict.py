import torch
from spimquant import create_model
from napari_cellseg3d.func_variants import normalize
from monai.inferers import sliding_window_inference
import numpy as np


# def main():
#     parser = argparse.ArgumentParser(description='Example with dictionary')
#     parser.add_argument('--input_np', type=str, default=None,
#                         help='Path to the input .npy image (3d or 4d, if 3d then no need to pass im_channel argument), '
#                              'this files should be created using np.save(input_image) with default arguments')
#     parser.add_argument('--im_channel', type=int, default=None,
#                         help='The channel of input 3d image to predict on. By default, assume the image '
#                              'does not have channel dimension (and is thus a single channel image)')
#     parser.add_argument('--roi_width', type=int, default=32,
#                         help='prediction roi width; the prediction will be tiling of 3d blocks of this size')
#     parser.add_argument('--result_np', type=str, default=None,
#                         help='path the result image file saves to')
#     parser.add_argument('--model_config', type=str, default=None,
#                         help='path to the saved CellSeg3DModelConfig json file')
#
#     args = parser.parse_args()
#     with open(args.model_config, 'r') as infile:
#         config: CellSeg3DModelConfig = json.load(infile, object_hook=strenc.get_decoder_hook())
#
#     im = np.load(args.input_np)
#     ch = args.im_channel
#     if ch is not None:
#         im = im[ch]
#     rw = args.roi_width
#     seg = inference_on_np3d(config, im, roi_size=(rw, rw, rw), model=None)
#
#     np.save(args.result_np, seg)


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


def inference_on_np3d(config: dict, im3d_np, roi_size, model=None):
    """
    im3d_np is 3d array convertible to float32, its dimensions are (z, y, x)
    """
    if model is None:
        model = create_model(config)
    return inference_on_np_batch(config, im3d_np[None, None], roi_size, model)
