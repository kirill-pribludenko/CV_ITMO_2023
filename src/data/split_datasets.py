import os
from typing import List
import random
import shutil

import rasterio as rio
from rasterio import windows

INPUT_IMG_PATH_G = "./data/interim/torchgeo/img/full_RGBN.tif"
INPUT_MASK_PATH_G = "./data/interim/torchgeo/mask/mask.tif"
OUTPUT_IMGS_DIR_G = [
    "./data/processed/torchgeo/img/train",
    "./data/processed/torchgeo/img/val",
]
OUTPUT_MASKS_DIR_G = [
    "./data/processed/torchgeo/mask/train",
    "./data/processed/torchgeo/mask/val",
]

INPUT_IMG_PATH_C = "./data/interim/classic/img"
INPUT_MASK_PATH_C = "./data/interim/classic/mask"
OUTPUT_IMGS_DIR_C = [
    "./data/processed/classic/img/train",
    "./data/processed/classic/img/val",
]
OUTPUT_MASKS_DIR_C = [
    "./data/processed/classic/mask/train",
    "./data/processed/classic/mask/val",
]

SCALE = 0.7


def split_dataset_torchgeo(
    input_path: str, output_path: List[str], file_name: str, ratio: float
) -> None:
    """
    Function split Big Sattelite Img or Mask for it to train & val part

    Args:
        input_path: path of file to split
        output_path: paths for saving img
        file_name: part of file name for saving img
        ratio: ratio according to which the split occurs

    Returns:
        None
    """
    with rio.open(input_path) as big_image:
        ncols, nrows = big_image.meta["width"], big_image.meta["height"]

        train_meta = big_image.meta.copy()
        val_meta = big_image.meta.copy()

        new_col = int(ncols * ratio)
        big_window = windows.Window(col_off=0, row_off=0,
                                    width=ncols, height=nrows)

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
        val_meta["width"], val_meta["height"] = (val_window.width,
                                                 val_window.height)
        val_outpath = os.path.join(output_path[1], f"{file_name}_val.tif")
        val_img = big_image.read(window=val_window)

        with rio.open(val_outpath, "w", **val_meta) as outds:
            outds.write(val_img)


def split_dataset_classic(i_input_dir: str, i_output_dir: List[str],
                          m_input_dir: str, m_output_dir: List[str],
                          ratio: float) -> None:
    """
    Splits a dataset into a train and a val set, and saves them in separate
    folders.

    Args:
        i_input_dir: path of img files to split.
        i_output_dir: paths for saving imgs.
        m_input_dir: path of mask files to split.
        m_output_dir: paths for saving masks.
        ratio: ratio according to which the split occurs

    Returns:
        None
    """
    # Get a list of all the files in the input directory
    file_list = os.listdir(i_input_dir)
    # Randomize the order of the files
    random.shuffle(file_list)
    i_file_list = file_list
    # Make m_file_list with the same order
    m_file_list = ['mask' + x[3:] for x in file_list]

    # Split the file list into training and validation sets
    split_idx = int(len(file_list) * ratio)
    i_train_files = i_file_list[:split_idx]
    i_val_files = i_file_list[split_idx:]
    m_train_files = m_file_list[:split_idx]
    m_val_files = m_file_list[split_idx:]

    # Copy the training files to the train directory
    for i_file_name, m_file_name in zip(i_train_files, m_train_files):
        i_src_path = os.path.join(i_input_dir, i_file_name)
        i_dst_path = os.path.join(i_output_dir[0], i_file_name)
        shutil.copy(i_src_path, i_dst_path)
        m_src_path = os.path.join(m_input_dir, m_file_name)
        m_dst_path = os.path.join(m_output_dir[0], m_file_name)
        shutil.copy(m_src_path, m_dst_path)

    # Copy the validation files to the val directory
    for i_file_name, m_file_name in zip(i_val_files, m_val_files):
        i_src_path = os.path.join(i_input_dir, i_file_name)
        i_dst_path = os.path.join(i_output_dir[1], i_file_name)
        shutil.copy(i_src_path, i_dst_path)
        m_src_path = os.path.join(m_input_dir, m_file_name)
        m_dst_path = os.path.join(m_output_dir[1], m_file_name)
        shutil.copy(m_src_path, m_dst_path)


def delete_images_not_512(folder: str) -> None:
    """
    Deletes all images in the given folder if the dimension of
    the image is not equal to 512 for each side.
    """
    for filename in os.listdir(folder):
        filepath = os.path.join(folder, filename)
        with rio.open(filepath) as img:
            # Check if the dimensions of the image are not equal to 512
            if img.width != 512 or img.height != 512:
                img.close()
                os.remove(filepath)
        img.close()


def main():
    # Spliting datasets
    split_dataset_torchgeo(INPUT_IMG_PATH_G, OUTPUT_IMGS_DIR_G,
                           "img", SCALE)
    split_dataset_torchgeo(INPUT_MASK_PATH_G, OUTPUT_MASKS_DIR_G,
                           "mask", SCALE)
    split_dataset_classic(INPUT_IMG_PATH_C, OUTPUT_IMGS_DIR_C,
                          INPUT_MASK_PATH_C, OUTPUT_MASKS_DIR_C, SCALE)

    # Delete all images which not equal 512
    folders_for_checking = OUTPUT_IMGS_DIR_C + OUTPUT_MASKS_DIR_C
    for folder in folders_for_checking:
        delete_images_not_512(folder)


if __name__ == "__main__":
    main()
