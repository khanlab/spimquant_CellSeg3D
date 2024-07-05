import numpy as np
from cvpl_tools.array_key_dict import ArrayKeyDict
import dask.array as da
import cvpl_tools.fs as fs
import cvpl_tools.dataset_reference as dataset_reference
import cvpl_tools.persistence as persistence
import copy


if __name__ == '__main__':  # Avoids the bug mentioned in https://github.com/snakemake/snakemake/issues/2678
    csconf = snakemake.params.cellsegment


"""--------------------------------Part 1: Implementation--------------------------------"""


def get_zarr_path():
    zarr_path = snakemake.params.zarr.replace('\\', '/')
    if zarr_path.startswith('gs:/') and zarr_path[4] != '/':
        zarr_path = "gs://" + zarr_path[4:]
    return zarr_path


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


def grab_splits_from_large_slice(
        zarr_subgroup,
        LD_SIZES: np.array,
        SPLIT_SIZES: np.array,
        IM_CHANNEL: int,
        out_dataset_dir: str,
        LOOP_PER_LD: int = None,
        slice_arr: ArrayKeyDict = None,
        MAX_N: int = None,
        CLIP_MIN: int = 0,
        CLIP_MAX: int = 1000,
        MIN_SLICE_VAR: float = 1000.,
        ld_start: np.array = None,  # starting point of sampling location
        pred_params: dict = None
) -> int:
    """
    Params
        zarr_subgroup - one large slice will be fetched from this location
        LD_SIZES - a len 3 array containing the size of the large slice
        SPLIT_SIZES - size of individual splits to be taken from the large slice
        LOOP_PER_LD - the number of times this function will try to take a split from the large slice; returned
            readed_splits is guaranteed to be smaller or equal to this number; if None, then ordered read over
            all splits in the slice are used instead
        pred_params - if not None, this is used to specify a model to provide corresponding masks to the images
    Return
        The number of files read
    """
    if slice_arr is None:
        slice_arr = ArrayKeyDict()

    fs.ensure_dir_exists(out_dataset_dir, True)

    LD_BY_SPLIT = LD_SIZES // SPLIT_SIZES
    zarr_shape = np.array(zarr_subgroup.shape, dtype=np.int32)
    LD_RNG = ((zarr_shape[1:] - 1) // LD_SIZES) + 1
    READ_N = 0
    im = np.zeros((2, LD_SIZES[0], LD_SIZES[1], LD_SIZES[2]), dtype=np.uint16)

    if ld_start is None:
        rng = np.random.randint(0, LD_RNG)
        ld_start = rng * LD_SIZES
    else:
        ld_start = np.array(ld_start)
    ld_end = ld_start + LD_SIZES
    result_im = da.from_zarr(zarr_subgroup)[:, ld_start[0]:ld_end[0], ld_start[1]:ld_end[1], ld_start[2]:ld_end[2]]
    result_im = result_im.compute()
    im[:result_im.shape[0], :result_im.shape[1], :result_im.shape[2], :result_im.shape[3]] = result_im
    print(f'Computed slice at location {ld_start}')
    if LOOP_PER_LD is None:
        def loop_over():
            for ind in np.argwhere(np.ones(LD_BY_SPLIT, dtype=np.uint8)):
                yield ind
    else:
        def loop_over():
            for i in range(LOOP_PER_LD):
                yield np.random.randint(0, LD_BY_SPLIT)

    if pred_params is not None:
        import predict
        model_config = persistence.read_dict(pred_params['model_config_path'])
        model = predict.create_model(model_config)
        pred_dataset_dir = pred_params['pred_dataset_dir']
        if 'slice_arr' in pred_params:
            pred_slice_arr = pred_params['slice_arr']
        else:
            pred_slice_arr = ArrayKeyDict()

    for rng2 in loop_over():
        split_start = rng2 * SPLIT_SIZES
        global_start = ld_start + split_start
        imid = f'{global_start[0].item()}_{global_start[1].item()}_{global_start[2].item()}'
        if imid in slice_arr:
            continue
        split_end = split_start + SPLIT_SIZES
        split = im[:, split_start[0]:split_end[0], split_start[1]:split_end[1], split_start[2]:split_end[2]]
        split = np.clip(split, CLIP_MIN, CLIP_MAX)
        if np.var(split[IM_CHANNEL]) < MIN_SLICE_VAR:  # too little variation, really nothing is here
            continue

        save_path = f'{out_dataset_dir}/{imid}.npy'
        slice_arr[imid] = dataset_reference.DatapointReference(save_path)
        READ_N += 1
        np.save(save_path, split)
        if pred_params is not None:
            output = predict.inference_on_np3d(model_config, split[IM_CHANNEL], [32, 32, 32], model=model)
            pred_save_path = f'{pred_dataset_dir}/{imid}.npy'
            np.save(pred_save_path, output)
            pred_slice_arr[imid] = pred_save_path
        if (MAX_N is not None) and (READ_N >= MAX_N):
            break
    return READ_N


def grab_splits_from_large_slices_as_dataset(
        dataset_name: str,
        dataset_info: str,
        zarr_subgroup,
        LD_SIZES: np.array,
        SPLIT_SIZES: np.array,
        IM_CHANNEL: int,
        out_dataset_dir: str,
        LOOP_PER_LD: int = None,
        TOTAL_N: int = None,
        CLIP_MIN: int = 0,
        CLIP_MAX: int = 1000,
        MIN_SLICE_VAR: float = 1000.,
        ld_start: np.array = None,  # starting point of sampling location
        pred_params: dict = None
):
    slice_arr = ArrayKeyDict()
    READ_N = 0
    if pred_params is not None:
        pred_params = copy.deepcopy(pred_params)
        pred_params['slice_arr'] = ArrayKeyDict()
    while True:
        splits_read = grab_splits_from_large_slice(
            zarr_subgroup=zarr_subgroup,
            LD_SIZES=LD_SIZES,
            SPLIT_SIZES=SPLIT_SIZES,
            LOOP_PER_LD=LOOP_PER_LD,
            IM_CHANNEL=IM_CHANNEL,
            out_dataset_dir=out_dataset_dir,
            slice_arr=slice_arr,
            MAX_N=TOTAL_N - READ_N,
            CLIP_MIN=CLIP_MIN,
            CLIP_MAX=CLIP_MAX,
            MIN_SLICE_VAR=MIN_SLICE_VAR,
            ld_start=ld_start
        )
        READ_N += splits_read
        print(f'Finished sampling splits from slice with READ_N={READ_N}')
        if READ_N >= TOTAL_N:
            break
    im_read_setting = fs.ImReadSetting(
        true_im_ndim=4,
        im_format=fs.ImFileType.FORMAT_UINT16)
    ref = dataset_reference.DatasetReference.new(
        slice_arr,
        dataset_name,
        dataset_info,
        im_read_setting
    )
    persistence.write_dataset_reference(ref, f'{out_dataset_dir}/DatasetReference')
    if pred_params is not None:
        im_read_setting = fs.ImReadSetting(
            true_im_ndim=3,
            im_format=fs.ImFileType.FORMAT_UINT16
        )  # instance segmentaion mask of given channel
        pred_ref = dataset_reference.DatasetReference.new(
            pred_params['slice_arr'],
            dataset_name,
            dataset_info,
            im_read_setting
        )
        persistence.write_dataset_reference(pred_ref,
                                            f'{pred_params["pred_dataset_dir"]}/DatasetReference')


"""--------------------------------Part 2: Interfaces-------------------------------------"""


def init_dataset(csconf):
    zarr_group = load_OME_ZARR_as_zarr_group(get_zarr_path())
    print(list(zarr_group.keys()))
    zarr_subgroup = zarr_group['0']
    zarr_shape = np.array(zarr_subgroup.shape, dtype=np.int32)
    print(zarr_shape)

    LD_SIZES = np.array(csconf["ld_size"], dtype=np.int32)
    SPLIT_SIZES = np.array(csconf["train_chunk_size"], dtype=np.int32)
    LOOP_PER_LD = csconf["num_chunk_per_ld"]  # from each LD slice, we attempt to read this many splits
    dataset_dir = snakemake.output.dataset_dir

    grab_splits_from_large_slices_as_dataset(
        dataset_name=csconf['dataset_name'],
        dataset_info=csconf["creation_info"],
        zarr_subgroup=zarr_subgroup,
        LD_SIZES=LD_SIZES,
        SPLIT_SIZES=SPLIT_SIZES,
        LOOP_PER_LD=LOOP_PER_LD,
        IM_CHANNEL=csconf['im_channel'],
        out_dataset_dir=dataset_dir,
        TOTAL_N=csconf["num_train_chunk"],
        CLIP_MIN=0,
        CLIP_MAX=1000,
        MIN_SLICE_VAR=1000
    )


def init_supervised_dataset(csconf):
    dataset_dir = snakemake.output.dataset_dir
    fs.ensure_dir_exists(dataset_dir, True)

    zarr_group = load_OME_ZARR_as_zarr_group(get_zarr_path())
    print(list(zarr_group.keys()))
    zarr_subgroup = zarr_group['0']
    zarr_shape = np.array(zarr_subgroup.shape, dtype=np.int32)
    print(zarr_shape)

    """
    Initialize the following datasets:
    1. Train set (random sample)
    2. Test set (random sample)
    3. Common region slice
    4. Olfactory bulb slice
    5. Edge slice
    """
    dataset_types = ('train_random_sample', 'test_random_sample', 'slice_common',
                     'slice_edge', 'slice_olfactory_bulb', 'slice_blurry')
    dataset_infos = (
        "Uniformly randomly sampled slices for training",
        "Uniformly randomly sampled slices for testing (same distribution as train_random_sample)",
        "A 1024 * 1024 slice of width 32 or 64; a regular brain region",
        "A 1024 * 1024 slice of width 32 or 64; located at the edge of the scanned brain",
        "A 1024 * 1024 slice of width 32 or 64; of the olfactory bulb",
        "A 1024 * 1024 slice of width 32 or 64; of a region that is imaged relatively blurry"
    )
    dataset_locs = (
        None,
        None,
        (1632, 4096, 2560),
        (1920, 2880, 4096),
        (224, 6592, 1856), # (1344, 512, 2048),
        (1344, 4608, 2048)
    )

    for sz, nsplits in ((32, 1024), (64, 256)):
        sz_prefix = f'sz{sz}'
        for im_channel in range(2):
            channel_str = f'ch{im_channel}'
            for i in range(len(dataset_types)):
                ds_type = dataset_types[i]
                ds_info = dataset_infos[i]
                ds_loc = dataset_locs[i]
                dataset_name = f'{sz_prefix}_{channel_str}_{ds_type}'
                im_dataset_dir = f'{dataset_dir}/{dataset_name}_im'
                pred_params = {
                    'model_config_path': snakemake.input.model_config,
                    'pred_dataset_dir': f'{dataset_dir}/{dataset_name}_cellseg3d'
                }

                SPLIT_SIZES = np.array((sz, ) * 3, dtype=np.int32)
                if ds_loc is None:
                    LOOP_PER_LD = sz // 16
                    LD_SIZES = np.array((sz, 2048, 2048), dtype=np.int32)
                else:
                    LOOP_PER_LD = None
                    LD_SIZES = np.array((sz, 1024, 1024), dtype=np.int32)

                grab_splits_from_large_slices_as_dataset(
                    dataset_name=dataset_name,
                    dataset_info=ds_info,
                    zarr_subgroup=zarr_subgroup,
                    LD_SIZES=LD_SIZES,
                    SPLIT_SIZES=SPLIT_SIZES,
                    LOOP_PER_LD=LOOP_PER_LD,
                    IM_CHANNEL=im_channel,
                    out_dataset_dir=im_dataset_dir,
                    TOTAL_N=nsplits,
                    CLIP_MIN=0,
                    CLIP_MAX=1000,
                    MIN_SLICE_VAR=1000,
                    ld_start=ds_loc,
                    pred_params=pred_params
                )


def main():
    init_supervised_dataset(csconf)


if __name__ == '__main__':
    main()
