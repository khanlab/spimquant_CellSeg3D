import os
import matplotlib.pyplot as plt
import numpy as np
import sys
import dask.array as da
import cvpl_tools.fs as fs
import cvpl_tools.dataset_reference as dataset_reference
from cvpl_tools.array_key_dict import ArrayKeyDict
import cvpl_tools.persistence as persistence


if __name__ == '__main__':  # Avoids the bug mentioned in https://github.com/snakemake/snakemake/issues/2678
    csconf = snakemake.params.cellsegment
    DATASET_NAME = csconf['dataset_name']
    IM_CHANNEL = csconf['im_channel']
    if 'device' in csconf:
        DEVICE = csconf['device']
        SCRATCH_FOLDER = csconf['scratch_dir']


def load_OME_ZARR_as_zarr_group(path: str):
    import zarr
    if path.strip().startswith('gs:'):  # google cloud storage:
        import gcsfs
        gfs = gcsfs.GCSFileSystem(token=None)
        store = gfs.get_mapper(path)
        zarr_group = zarr.open(store, mode='r')
    else:  # local file system
        zarr_group = zarr.open(path, 'r')
    return zarr_group


def main():
    import sys
    sys.path.append('../../..//spimquant/submodules/spimquant_cellpose')
    cmd = snakemake.params.command
    if cmd == 'init_dataset':
        init_dataset()
    elif cmd == 'train':
        train()
    elif cmd == 'predict':
        inference()
    else:
        raise ValueError(f'ERROR: Snakemake command is not recognized: {snakemake.command}')


def init_dataset():
    dataset_dir = snakemake.output.dataset_dir
    fs.ensure_dir_exists(dataset_dir, True)
    TOTAL_N = csconf["num_train_chunk"]  # total number of splits we want in the end
    CREATION_INFO = csconf["creation_info"]
    zarr_group = load_OME_ZARR_as_zarr_group(csconf['zarr'])
    print(list(zarr_group.keys()))
    zarr_subgroup = zarr_group['0']

    zarr_shape = np.array(zarr_subgroup.shape, dtype=np.int32)
    print(zarr_shape)
    LD_SIZES = np.array(csconf["ld_size"], dtype=np.int32)
    SPLIT_SIZES = np.array(csconf["train_chunk_size"], dtype=np.int32)
    LD_BY_SPLIT = LD_SIZES // SPLIT_SIZES
    LOOP_PER_LD = csconf["num_chunk_per_ld"]  # from each LD slice, we attempt to read this many splits

    slice_arr = ArrayKeyDict()
    LD_RNG = ((zarr_shape[1:] - 1) // LD_SIZES) + 1
    READ_N = 0
    im = np.zeros((2, LD_SIZES[0], LD_SIZES[1], LD_SIZES[2]), dtype=np.uint16)
    while True:
        rng = np.random.randint(0, LD_RNG)
        ld_start = rng * LD_SIZES
        ld_end = ld_start + LD_SIZES
        result_im = da.from_zarr(zarr_subgroup)[:, ld_start[0]:ld_end[0], ld_start[1]:ld_end[1], ld_start[2]:ld_end[2]]
        result_im = result_im.compute()
        im[:result_im.shape[0], :result_im.shape[1], :result_im.shape[2], :result_im.shape[3]] = result_im
        print(f'Computed slice at location {ld_start}')
        for i in range(LOOP_PER_LD):
            rng2 = np.random.randint(0, LD_BY_SPLIT)
            split_start = rng2 * SPLIT_SIZES
            global_start = ld_start + split_start
            imid = f'{global_start[0].item()}_{global_start[1].item()}_{global_start[2].item()}'
            if imid in slice_arr:
                continue
            split_end = split_start + SPLIT_SIZES
            split = im[:, split_start[0]:split_end[0], split_start[1]:split_end[1], split_start[2]:split_end[2]]
            split = np.clip(split, 0, 1000)
            if np.var(split[IM_CHANNEL]) < 1000.:  # too little variation, really nothing is here
                continue

            save_path = f'{dataset_dir}/{imid}.npy'
            slice_arr[imid] = dataset_reference.DatapointReference(save_path)
            READ_N += 1
            np.save(save_path, split)
            if READ_N >= TOTAL_N:
                break
        print(f'Finished sampling splits from slice with READ_N={READ_N}')
        if READ_N >= TOTAL_N:
            break
    im_read_setting = fs.ImReadSetting(
        true_im_ndim=4,
        im_format=fs.ImFileType.FORMAT_UINT16)
    ref = dataset_reference.DatasetReference.new(
        slice_arr,
        DATASET_NAME,
        CREATION_INFO,
        im_read_setting
    )
    persistence.write_dataset_reference(ref, f'{dataset_dir}/DatasetReference')


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
    loc = csconf['predict_range']
    zarr_group = load_OME_ZARR_as_zarr_group(csconf['zarr'])
    zarr_subgroup = zarr_group['0']
    im = da.from_zarr(zarr_subgroup)[IM_CHANNEL, loc[0][0]:loc[0][1],
                loc[1][0]:loc[1][1], loc[2][0]:loc[2][1]]

    model_config = persistence.read_dict(snakemake.input.model_config)
    output = predict.inference_on_np3d(model_config, im, [32, 32, 32])
    print('finished inference')
    output = output.detach().cpu().numpy()[0]
    print('output extracted')
    output_argmax = output.argmax(axis=0)
    print('argmax computed')
    result_dir = snakemake.output.pred_result_dir
    fs.ensure_dir_exists(result_dir, True)
    np.save(f'{result_dir}/inference.npy', output)
    print('output written')
    np.save(f'{result_dir}/inference_argmax.npy', output_argmax)
    print('argmax written')
    np.save(f'{result_dir}/inference_im.npy', im)
    print('inference image written')


if __name__ == '__main__':
    main()

