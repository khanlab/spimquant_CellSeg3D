import os
import matplotlib.pyplot as plt
import numpy as np
import sys
import dask.array as da
import cvpl_tools.fs as fs
import cvpl_tools.dataset_reference as dataset_reference
from cvpl_tools.array_key_dict import ArrayKeyDict
import cvpl_tools.persistence as persistence
from supervised_dataset_gen import init_dataset, get_zarr_path, load_OME_ZARR_as_zarr_group


if __name__ == '__main__':  # Avoids the bug mentioned in https://github.com/snakemake/snakemake/issues/2678
    csconf = snakemake.params.cellsegment
    DATASET_NAME = csconf['dataset_name']
    IM_CHANNEL = csconf['im_channel']
    if 'device' in csconf:
        DEVICE = csconf['device']
        SCRATCH_FOLDER = csconf['scratch_dir']


def main():
    import sys
    cmd = snakemake.params.command
    if cmd == 'init_dataset':
        init_dataset(csconf, snakemake=snakemake)
    elif cmd == 'train':
        train()
    elif cmd == 'predict':
        inference()
    else:
        raise ValueError(f'ERROR: Snakemake command is not recognized: {snakemake.command}')


def train():
    import logging
    from napari_cellseg3d.dev_scripts.colab_training import logger
    import train
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    logger.addHandler(handler)

    dataset_dir = snakemake.input.dataset_dir
    trainset = persistence.read_dataset_reference(f'{dataset_dir}/DatasetReference')
    model_weight_dir = os.path.abspath(f'{snakemake.params.output_dir}/{DATASET_NAME}_model_weight')
    fs.ensure_dir_exists(model_weight_dir, True)
    scratch_dir = f'{SCRATCH_FOLDER}/{DATASET_NAME}'
    fs.ensure_dir_exists(scratch_dir, True)
    config = {
        'trainset': trainset,
        'scratch_folder': scratch_dir,
        'result_folder': model_weight_dir,
        'im_channel': IM_CHANNEL,
        'num_classes': csconf['num_classes'],
        'input_brightness_range': (csconf['min_brightness'], csconf['max_brightness']),
        'nepochs': csconf['nepochs'],
        'n_cuts_weight': csconf['n_cuts_weight'],
        'rec_loss_weight': csconf['rec_loss_weight'],
        'device': csconf['device'],
        'in_channels': 1,
        'out_channels': 1,
        'dropout': .65,
        'rec_loss': 'MSE',
        'model_weight_path': None,
        'post_processing_code': None,
    }

    config = train.train_model(config)
    persistence.write_dict(config, snakemake.output.model_config)


def inference():
    import predict
    locs = csconf['predict_range']
    zarr_group = load_OME_ZARR_as_zarr_group(get_zarr_path())
    zarr_subgroup = zarr_group['0']

    model_config = persistence.read_dict(snakemake.input.model_config)
    result_dir = snakemake.output.pred_result_dir
    fs.ensure_dir_exists(result_dir, True)
    for i in range(len(locs)):
        loc = locs[i]
        im = da.from_zarr(zarr_subgroup)[IM_CHANNEL, loc[0][0]:loc[0][1],
                    loc[1][0]:loc[1][1], loc[2][0]:loc[2][1]]

        output = predict.inference_on_np3d(model_config, im, [32, 32, 32])
        print('finished inference')
        output = output.detach().cpu().numpy()[0]
        print('output extracted')
        output_argmax = output.argmax(axis=0)
        print('argmax computed')
        np.save(f'{result_dir}/inference_{i}.npy', output)
        print('output written')
        np.save(f'{result_dir}/inference_argmax_{i}.npy', output_argmax)
        print('argmax written')
        np.save(f'{result_dir}/inference_im_{i}.npy', im)
        print('inference image written')


if __name__ == '__main__':
    main()

