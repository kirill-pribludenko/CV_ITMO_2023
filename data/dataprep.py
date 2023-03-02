import os
import glob
import time
import rasterio as rio
import numpy as np
from rasterio import windows
from itertools import product

SAT_DIR = '''./data/raw/S2A_MSIL2A_20220703T084611_N0400_R107_T37VCD_20220703T134122\
/S2A_MSIL2A_20220703T084611_N0400_R107_T37VCD_20220703T134122.SAFE\
/GRANULE/L2A_T37VCD_A036712_20220703T085107/IMG_DATA/R10m/'''
IMGS_DIR = "./data/raw/RGB"
OUTPUT_IMGS_DIR = "./data/output/img"
OUTPUT_IMGS_FN = 'img'
MASKS_DIR = "./data/raw/mask"
OUTPUT_MASKS_DIR = "./data/output/mask"
OUTPUT_MASKS_FN = 'mask'
IMAGE_SIZE = 512


def create_rgb_img_and_mask(satpath):
    '''
    Function create RGB image and mask from raw satellite
    image also do simple correction of brightness, gamma
    and normalize values of img
    '''
    # read blue
    band2 = rio.open(satpath + 'T37VCD_20220703T084611_B02_10m.jp2',
                     driver='JP2OpenJPEG')
    # read green
    band3 = rio.open(satpath + 'T37VCD_20220703T084611_B03_10m.jp2',
                     driver='JP2OpenJPEG')
    # read red
    band4 = rio.open(satpath + 'T37VCD_20220703T084611_B04_10m.jp2',
                     driver='JP2OpenJPEG')
    # read nir
    band8 = rio.open(satpath + 'T37VCD_20220703T084611_B08_10m.jp2',
                     driver='JP2OpenJPEG')
    # will be teplate for masks with right crs
    mask = rio.open(satpath + 'T37VCD_20220703T084611_B08_10m.jp2',
                    driver='JP2OpenJPEG')

    def brighten_gammacor(band):
        alpha = 0.1
        beta = 0.5
        gamma = 2
        band = np.clip(alpha*band+beta, 0, 255)
        return np.power(band, 1/gamma)

    def normalize(band):
        return (255*((band-band.min())/(band.max() - band.min()))).astype(int)

    red_bg = brighten_gammacor(band4.read(1))
    blue_bg = brighten_gammacor(band2.read(1))
    green_bg = brighten_gammacor(band3.read(1))

    red_bgn = normalize(red_bg)
    green_bgn = normalize(green_bg)
    blue_bgn = normalize(blue_bg)

    # save RGB color image
    trueColor = rio.open('./data/raw/RGB/Sentinel_RGB.tif', 'w', driver='Gtiff',
                         width=band8.width, height=band8.height,
                         count=3,
                         crs=band8.crs,
                         transform=band8.transform,
                         dtype=np.uint8
                         )
    trueColor.write(red_bgn, 1)
    trueColor.write(green_bgn, 2)
    trueColor.write(blue_bgn, 3)
    trueColor.close()

    # save template of mask image
    mask = mask.read(1)
    mask[:, :] = 0
    mask_img = rio.open('./data/raw/mask/Sentinel_RGB.tif', 'w', driver='Gtiff',
                        width=band8.width, height=band8.height,
                        count=1,
                        crs=band8.crs,
                        transform=band8.transform,
                        dtype=np.uint8
                        )
    mask_img.write(mask, 1)
    mask_img.close()


def split_images(in_path, out_path, output_filename, img_size=256):
    '''
    A function to split the large images into squared images of
    size equal to TARGET_SIZE. Stores the new images into
    a directory named output, located data directory.
    '''
    def get_tiles(big_image, width=img_size, height=img_size):

        ncols, nrows = big_image.meta['width'], big_image.meta['height']
        offsets = product(range(0, ncols, width), range(0, nrows, height))
        big_window = windows.Window(col_off=0, row_off=0, width=ncols, height=nrows)
        for col_off, row_off in offsets:
            window = windows.Window(col_off=col_off, row_off=row_off, width=width, height=height).intersection(big_window)
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
                meta['transform'] = transform
                meta['width'], meta['height'] = window.width, window.height
                outpath = os.path.join(out_path, output_filename + f'_{int(window.col_off)}_{window.row_off}.tif')
                with rio.open(outpath, 'w', **meta) as outds:
                    outds.write(big_image.read(window=window))

    print(f"Processed {img_filename} {i + 1}/{len(img_paths)}")
    mins, sec = divmod(time.time()-tic, 60)
    print(f"Execution completed in {mins} minutes and {sec:.2f} seconds.")


# create img and mask
create_rgb_img_and_mask(SAT_DIR)
# split img
split_images(IMGS_DIR, OUTPUT_IMGS_DIR, OUTPUT_IMGS_FN, img_size=IMAGE_SIZE)
# split mask
split_images(MASKS_DIR, OUTPUT_MASKS_DIR, OUTPUT_MASKS_FN, img_size=IMAGE_SIZE)
