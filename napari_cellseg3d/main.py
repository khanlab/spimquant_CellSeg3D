import numpy as np
import os
import json
import cvpl_tools.fs as fs
from pathlib import Path
import shutil
import scipy.ndimage as ndimage
import skimage
import skimage.morphology as morph
import sys
import cvpl_tools.persistence as persistence
from cvpl_tools.np_algs import CountFromMask_Watershed, NCellFromInst_BySize
from napari_cellseg3d.func_variants import normalize
import random


def main():
    print(sys.path)
    import predict

    model_config = persistence.read_dict('D:/progtools/RobartsResearch/data/lightsheet/mousebrain_chan0_20240627_1_model_config')
    model = predict.create_model(model_config)

    ncell_from_inst = NCellFromInst_BySize(30., 220.)
    to_inst = CountFromMask_Watershed(ncell_from_inst,
                                      size_thres=60.,
                                      dist_thres=1.,
                                      rst=None,
                                      size_thres2=100.,
                                      dist_thres2=1.5,
                                      rst2=60.)

    IN_PATH = 'D:/progtools/RobartsResearch/SPIMquant/dataset/supervised_datasets'
    OUT_PATH = 'D:/progtools/RobartsResearch/SPIMquant/dataset/supervised_datasets_watershed_json'
    fs.ensure_dir_exists(OUT_PATH, True)
    for folder_name in os.listdir(IN_PATH):
        folder_full_path = Path(f'{IN_PATH}/{folder_name}')
        if not os.path.isdir(folder_full_path):
            continue
        out_folder_path = Path(f'{OUT_PATH}/{folder_name}')
        out_folder_path_seg = Path(f'{OUT_PATH}/{folder_name}'[:-3] + '_cellseg3d')
        print(f'Creating folder {out_folder_path.resolve()}')
        fs.ensure_dir_exists(out_folder_path, True)
        print(f'Creating folder {out_folder_path_seg.resolve()}')
        fs.ensure_dir_exists(out_folder_path_seg, True)
        shutil.copytree(folder_full_path / 'DatasetReference', out_folder_path / 'DatasetReference')
        for im_name in os.listdir(folder_full_path):
            im_full_path = Path(f'{folder_full_path.resolve()}/{im_name}')
            if im_full_path.suffix != '.npy':
                continue
            # print(im_name)
            im: np.array = np.load(im_full_path)
            ch = 0 if 'ch0' in folder_name else 1
            im = im[ch]

            sigma = .62
            if ch == 0:
                thres = 410.
            else:
                thres = 310.
            im = ndimage.gaussian_filter(im, sigma=sigma, mode='nearest')
            seg_bin = to_inst.inst(im > thres)

            out_im_path = f'{out_folder_path.resolve()}/{im_full_path.stem}.json'
            out_seg_path = (f'{out_folder_path_seg.resolve()}/'
                            f'{im_full_path.stem}.json')
            # np.save(out_im_path, im)
            # np.save(out_seg_path, seg_bin)
            with open(out_im_path, 'w') as outfile:
                json.dump(im.tolist(), outfile)
            with open(out_seg_path, 'w') as outfile:
                json.dump(seg_bin.tolist(), outfile)


if __name__ == '__main__':
    main()
