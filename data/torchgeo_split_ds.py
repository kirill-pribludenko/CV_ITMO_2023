import os
from typing import List

import rasterio as rio
from rasterio import windows

INPUT_IMG_PATH = "./data/output/for_torchgeo_way/img/Sentinel_RGBN.tif"
INPUT_MASK_PATH = "./data/output/for_torchgeo_way/mask/mask.tif"
OUTPUT_IMGS_DIR = [
    "./data/output/for_torchgeo_way/img_f/train",
    "./data/output/for_torchgeo_way/img_f/val",
]
OUTPUT_MASKS_DIR = [
    "./data/output/for_torchgeo_way/mask_f/train",
    "./data/output/for_torchgeo_way/mask_f/val",
]
SCALE = 0.7


def train_test_split(
    input_path: str, output_path: List(str), file_name: str, ratio: float
) -> None:
    """
    Function split Big Sattelite Img or Mask for it to train & val part

    Args:
        input_path: path of file to split
        output_path: paths for saving imgs
        file_name: part of file name for saving imgs
        ratio: ratio according to which the split occurs

    Returns:
        None
    """
    with rio.open(input_path) as big_image:
        ncols, nrows = big_image.meta["width"], big_image.meta["height"]

        train_meta = big_image.meta.copy()
        val_meta = big_image.meta.copy()

        new_col = int(ncols * ratio)
        big_window = windows.Window(col_off=0, row_off=0, width=ncols, height=nrows)

        train_window = windows.Window(
            col_off=0, row_off=0, width=new_col, height=nrows
        ).intersection(big_window)
        train_transform = windows.transform(train_window, big_image.transform)
        train_meta["transform"] = train_transform
        train_meta["width"], train_meta["height"] = (
            train_window.width,
            train_window.height,
        )
        train_outpath = os.path.join(output_path[0], f"{file_name}_train.tif")
        train_img = big_image.read(window=train_window)

        with rio.open(train_outpath, "w", **train_meta) as outds:
            outds.write(train_img)

        val_window = windows.Window(
            col_off=new_col, row_off=0, width=ncols, height=nrows
        ).intersection(big_window)
        val_transform = windows.transform(val_window, big_image.transform)
        val_meta["transform"] = val_transform
        val_meta["width"], val_meta["height"] = val_window.width, val_window.height
        val_outpath = os.path.join(output_path[1], f"{file_name}_val.tif")
        val_img = big_image.read(window=val_window)

        with rio.open(val_outpath, "w", **val_meta) as outds:
            outds.write(val_img)


def main():
    train_test_split(INPUT_IMG_PATH, OUTPUT_IMGS_DIR, "img", SCALE)
    train_test_split(INPUT_MASK_PATH, OUTPUT_MASKS_DIR, "mask", SCALE)


if __name__ == "__main__":
    main()
