import numpy as np
from PIL import Image
import os
import cv2
from dask_image.imread import imread

input_train_vol_path = "/Users/vmax/Documents/TRAILMAP/data/training/training-original/volumes"
input_train_label_path = "/Users/vmax/Documents/TRAILMAP/data/training/training-original/labels"
output_train_vol_path = "/Users/vmax/Documents/3dstardist/training/training_source"
output_train_label_path = "/Users/vmax/Documents/3dstardist/training/training_target"

input_qcvisual_label_path = "/Users/vmax/Documents/3d/Annotation_2021/visual_area/experimentC_sample1/labels.tif"
input_qcvisual_vol_path = "/Users/vmax/Documents/3d/Annotation_2021/visual_area/experimentC_sample1/images.tif"
input_qcsm_label_path = "/Users/vmax/Documents/TRAILMAP/data/validation/validation-original/oldlabels/c5labels.tif"
input_qcsm_vol_path = "/Users/vmax/Documents/TRAILMAP/data/validation/validation-original/volumes/c5images.tif"

output_qcvisual_label_path = "/Users/vmax/Documents/3dstardist/qc/visual/target"
output_qcvisual_vol_path = "/Users/vmax/Documents/3dstardist/qc/visual/source"
output_qcsm_label_path = "/Users/vmax/Documents/3dstardist/qc/sm/target"
output_qcsm_vol_path = "/Users/vmax/Documents/3dstardist/qc/sm/source"

os.makedirs(output_train_label_path, exist_ok=True)
os.makedirs(output_train_vol_path, exist_ok=True)
os.makedirs(output_qcvisual_label_path, exist_ok=True)
os.makedirs(output_qcvisual_vol_path, exist_ok=True)
os.makedirs(output_qcsm_label_path, exist_ok=True)
os.makedirs(output_qcsm_vol_path, exist_ok=True)

j = 0
for filename in sorted(os.listdir(input_train_label_path)):
    if filename.endswith(".tif"):
        img = imread(os.path.join(input_train_label_path, filename))
        slice = img.compute().astype(np.uint8)
        im = Image.fromarray(slice[0])
        ims = []
        for i in range(1, slice.shape[0]):
            ims.append(Image.fromarray(slice[i]))
        im.save(f"{output_train_label_path}/img_{str(j).zfill(4)}.tif", save_all=True, append_images=ims)
        j += 1
j = 0
for filename in sorted(os.listdir(input_train_vol_path)):
    if filename.endswith(".tif"):
        img = Image.open(os.path.join(input_train_vol_path, filename))
        img.save(f"{output_train_vol_path}/img_{str(j).zfill(4)}.tif", save_all=True)
        j += 1

img1, img2 = imread(input_qcsm_label_path), imread(input_qcvisual_label_path)

slice = img1.compute().astype(np.uint8)
im = Image.fromarray(slice[0])
ims = []
for i in range(1, slice.shape[0]):
    ims.append(Image.fromarray(slice[i]))
im.save(f"{output_qcsm_label_path}/img_{str(1).zfill(4)}.tif", save_all=True, append_images=ims)

slice = img2.compute().astype(np.uint8)
im = Image.fromarray(slice[0])
ims = []
for i in range(1, slice.shape[0]):
    ims.append(Image.fromarray(slice[i]))
im.save(f"{output_qcvisual_label_path}/img_{str(1).zfill(4)}.tif", save_all=True, append_images=ims)

img1, img2 = Image.open(input_qcsm_vol_path), Image.open(input_qcvisual_vol_path)
img1.save(f"{output_qcsm_vol_path}/img_{str(1).zfill(4)}.tif", save_all=True)
img2.save(f"{output_qcvisual_vol_path}/img_{str(1).zfill(4)}.tif", save_all=True)