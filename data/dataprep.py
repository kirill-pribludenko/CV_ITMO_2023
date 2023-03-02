import os
import glob
import time
import rasterio as rio
from rasterio import windows
from itertools import product


IMGS_DIR = "./data/raw/RGB"
OUTPUT_IMGS_DIR = "./data/output/img"
OUTPUT_IMGS_FN = 'img'
MASKS_DIR = "./data/raw/mask"
OUTPUT_MASKS_DIR = "./data/output/mask"
OUTPUT_MASKS_FN = 'mask'
IMAGE_SIZE = 512


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


# split img
split_images(IMGS_DIR, OUTPUT_IMGS_DIR, OUTPUT_IMGS_FN, img_size=IMAGE_SIZE)
# split mask
split_images(MASKS_DIR, OUTPUT_MASKS_DIR, OUTPUT_MASKS_FN, img_size=IMAGE_SIZE)

#TODO
# сделать обработку изображения