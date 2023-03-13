import glob
import os
import time
from itertools import product
from typing import List

import numpy as np
import rasterio as rio
from rasterio import windows

SAT_DIR = """./data/raw/S2A_MSIL2A_20220703T084611_N0400_R107_T37VCD_20220703T134122\
/S2A_MSIL2A_20220703T084611_N0400_R107_T37VCD_20220703T134122.SAFE\
/GRANULE/L2A_T37VCD_A036712_20220703T085107/IMG_DATA/R10m/"""
OUTPUT_IMGS_DIR = "./data/output/for_classic_way/img"
OUTPUT_MASKS_DIR = "./data/output/for_classic_way/mask_template"
LIST_OUTPUT_PATH = ["./data/raw/RGB", "./data/raw/RGBN", "./data/raw/mask_template"]
LIST_OUTPUT_FN = ["/Sentinel_RGB.tif", "/Sentinel_RGBN.tif", "/Sentinel_MASK.tif"]
IMAGE_SIZE = 512


def create_imgs_and_mask(
    sat_path: str, list_output_path: List[str], list_of_fn: List[str]
) -> None:
    """Function create RGB, RGBN images and draft of mask
    from raw data downloded from satellite. With images also
    do simple correction of brightness, gamma and normalize
    values of imgs.

    Args:
        sat_path: path to raw data from satellite
        list_output_path: list of paths where will be saved images
        list_of_fn: list of saved file names

    Returns:
        None
    """

    def save_img(
        output_path: str,
        output_filename: str,
        list_of_bands: List[np.ndarray],
        band_orig: np.ndarray,
    ) -> None:
        """Saving img via lib rasterio.
        IMPORTANT to give func the original band because it contains crs

        Args:
            output_path: path for saving
            output_filename: file name for saving
            list_of_bands: list of bands which will be saving to img
            band_orig: original band for saving crs

        Returns:
            None
        """

        img = rio.open(
            (output_path + output_filename),
            "w",
            driver="Gtiff",
            width=band_orig.width,
            height=band_orig.height,
            count=len(list_of_bands),
            crs=band_orig.crs,
            transform=band_orig.transform,
            dtype=np.uint16,
        )

        for i, band in zip(range(1, len(list_of_bands) + 1), list_of_bands):
            img.write(band, i)

        img.close()

    def simple_transformation(band: np.ndarray) -> np.ndarray:
        """Add brightness, gamma correction and normalization for band.

        Args:
            band: input band for transformation

        Returns:
            transformed band
        """

        # brightness
        alpha = 0.09
        beta = 0.4
        band = np.clip(alpha * band + beta, 0, 255)

        # gamma correction
        # gamma = 2
        # band = np.power(band, 1/gamma)

        # normalization
        band = 255 * ((band - band.min()) / (band.max() - band.min()))

        return band.astype(int)

    tic = time.time()
    print("Start creating the images...\n")
    # read blue
    band2 = rio.open(
        sat_path + "T37VCD_20220703T084611_B02_10m.jp2", driver="JP2OpenJPEG"
    )
    # read green
    band3 = rio.open(
        sat_path + "T37VCD_20220703T084611_B03_10m.jp2", driver="JP2OpenJPEG"
    )
    # read red
    band4 = rio.open(
        sat_path + "T37VCD_20220703T084611_B04_10m.jp2", driver="JP2OpenJPEG"
    )
    # read nir
    band8 = rio.open(
        sat_path + "T37VCD_20220703T084611_B08_10m.jp2", driver="JP2OpenJPEG"
    )
    # will be teplate for masks with right crs
    mask = rio.open(
        sat_path + "T37VCD_20220703T084611_B08_10m.jp2", driver="JP2OpenJPEG"
    )

    # transform to np.ndarrays
    list_bands_rgb = [band4.read(1), band3.read(1), band2.read(1)]
    list_bands_rgbn = [band4.read(1), band3.read(1), band2.read(1), band8.read(1)]
    mask = mask.read(1)
    mask[:, :] = 0  # black mask

    # apply func simple_transformation
    list_bands_rgb = [simple_transformation(band) for band in list_bands_rgb]
    list_bands_rgbn = [simple_transformation(band) for band in list_bands_rgbn]
    # mask is black, that way no need to apply func above
    list_bands_mask = [mask]

    # saving RGB, RGBN and mask
    save_img(list_output_path[0], list_of_fn[0], list_bands_rgb, band8)
    save_img(list_output_path[1], list_of_fn[1], list_bands_rgbn, band8)
    save_img(list_output_path[2], list_of_fn[2], list_bands_mask, band8)

    mins, sec = divmod(time.time() - tic, 60)
    print(f"Creating completed in {mins} minutes and {sec:.2f} seconds.")


def split_images(
    in_path: str, out_path: str, output_filename: str, img_size: int = 256
) -> None:
    """A function split the large images into squared images of
    size equal to img_size. Stores the new images into
    a directory named output, located data directory.

    Args:
        in_path: input path of files to split
        out_path: output path
        output_filename: part of output file name
        img_size: output size images

    Returns:
        None
    """

    def get_tiles(big_image, width=img_size, height=img_size):
        ncols, nrows = big_image.meta["width"], big_image.meta["height"]
        offsets = product(range(0, ncols, width), range(0, nrows, height))
        big_window = windows.Window(col_off=0, row_off=0, width=ncols, height=nrows)
        for col_off, row_off in offsets:
            window = windows.Window(
                col_off=col_off, row_off=row_off, width=width, height=height
            ).intersection(big_window)
            transform = windows.transform(window, big_image.transform)
            yield window, transform

    tic = time.time()
    print("Splitting the images...\n")
    img_paths = glob.glob(os.path.join(in_path, "*.tif"))
    img_paths.sort()

    for i, img_path in enumerate(img_paths):
        img_filename = os.path.join(os.path.basename(img_path))
        print(img_filename)

        with rio.open(img_path) as big_image:
            meta = big_image.meta.copy()

            for window, transform in get_tiles(big_image):
                meta["transform"] = transform
                meta["width"], meta["height"] = window.width, window.height
                outpath = os.path.join(
                    out_path,
                    output_filename + f"_{int(window.col_off)}_{window.row_off}.tif",
                )
                with rio.open(outpath, "w", **meta) as outds:
                    outds.write(big_image.read(window=window))

    print(f"Processed {img_filename} {i + 1}/{len(img_paths)}")
    mins, sec = divmod(time.time() - tic, 60)
    print(f"Execution completed in {mins} minutes and {sec:.2f} seconds.")


def main():
    # create imgs and mask
    create_imgs_and_mask(SAT_DIR, LIST_OUTPUT_PATH, LIST_OUTPUT_FN)
    # split img
    split_images(LIST_OUTPUT_PATH[0], OUTPUT_IMGS_DIR, "img", img_size=IMAGE_SIZE)
    # split mask
    split_images(LIST_OUTPUT_PATH[2], OUTPUT_MASKS_DIR, "mask", img_size=IMAGE_SIZE)


if __name__ == "__main__":
    main()
