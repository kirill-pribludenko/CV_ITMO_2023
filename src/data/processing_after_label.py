import glob
import os
import xml.etree.ElementTree as ET
from typing import List

import cv2
import numpy as np
import rasterio as rio
from rasterio.merge import merge

LABEL_FILE_NAME = "./data/interim/annotations.xml"
INPUT_MASKS_DIR = "./data/interim/classic/mask_template/"
OUTPUT_PATH = "./data/interim/classic/mask/"
OUTPUT_FILE = "./data/interim/torchgeo/mask/mask.tif"


def rle_decode(input_list: List[int], size_of_mask: List[int]) -> np.ndarray:
    """A function decore rle coding of polygons mask
    from CVAT format to np.ndarray.
    Mask - white ccolor, field - black

    Args:
        input_list: input list for decoding
        size_of_mask: size of mask

    Returns:
        decoding np.ndarray
    """

    new_str = ""
    new_size = [size + 1 for size in size_of_mask]

    for i, number in enumerate(input_list):
        condition = i % 2
        if condition == 0:
            new_str += "0 " * number
        elif condition == 1:
            new_str += "255 " * number

    new_array = np.fromstring(new_str, dtype=int, sep=" ")

    return new_array.reshape(new_size)


def main():
    """Reading xml file with coordinates of polygon mask
    for each file and generate dict with coord of all masks
    Structure of dict
    {'name_of_file_N': {'polygon_N': '[array_points]'}
    } - if polygons exist
    {'name_of_file_N': False} - if polygons not exist
    """

    print("Start reading XML file....")
    tree = ET.parse(LABEL_FILE_NAME)
    root = tree.getroot()
    dict_masks = {}

    # Opening XML file and create dict ---------------------------------------
    for i in range(2, len(root)):
        # changing name from img_* to mask_* for dict
        correct_name = root[i].attrib["name"]
        correct_name = "mask" + correct_name[3:]

        if len(root[i]) > 0:
            dict_temp = {}
            for j in range(len(root[i])):
                tag = root[i][j].tag
                if tag == "polygon":
                    # parse data from polygon
                    text_coords = root[i][j].attrib["points"].replace(";", ",")
                    array_points = (np.array(text_coords.split(","))
                                      .astype(float))
                    array_points = array_points.reshape(-1, 2).astype(int)
                    dict_temp[f"polygon_{j}"] = array_points
                elif tag == "mask":
                    # parse data from mask
                    text_coords = root[i][j].attrib["rle"]
                    # top - row, left - column
                    # height - row, width - column
                    left_top_point = [
                        int(root[i][j].attrib["top"]),
                        int(root[i][j].attrib["left"]),
                    ]
                    size_of_mask = [
                        int(root[i][j].attrib["height"]),
                        int(root[i][j].attrib["width"]),
                    ]
                    array_np = (np.fromstring(text_coords, dtype=int, sep=",")
                                  .tolist())
                    decoded_array = rle_decode(array_np, size_of_mask)
                    dict_temp[f"mask_{j}"] = {
                        "array": decoded_array,
                        "left_top_point": left_top_point,
                    }
                else:
                    print("Error of type annotation")

            dict_masks[correct_name] = dict_temp
        else:
            dict_masks[correct_name] = False

    # Fill all polygons in templates masks -----------------------------------
    print("Start creating masks....")
    for file_name in dict_masks.keys():
        # open file and transform it to array
        mask_orig = rio.open(INPUT_MASKS_DIR + file_name)
        img_array = mask_orig.read(1)

        # for case when no label, save a orig_file in right folder
        if dict_masks[file_name] is False:
            with rio.open(
                (OUTPUT_PATH + file_name),
                "w",
                driver="GTiff",
                dtype=rio.uint8,
                count=1,
                width=mask_orig.width,
                height=mask_orig.height,
                transform=mask_orig.transform,
            ) as dst:
                dst.write(img_array, indexes=1)
        # for case when label exist, fill all polygons and save file
        else:
            final_mask = np.zeros(img_array.shape)
            mask_m = np.zeros(img_array.shape)
            for i, key in enumerate(dict_masks[file_name].keys()):
                if key[:1] == "p":
                    polygon_array = dict_masks[file_name][key]
                    mask_p = cv2.fillPoly(img_array,
                                          pts=[polygon_array],
                                          color=(255))
                    final_mask += mask_p

                elif key[:1] == "m":
                    mask_array = dict_masks[file_name][key]["array"]
                    left_top_point = (dict_masks[file_name][key]
                                      ["left_top_point"])
                    from_row = left_top_point[0]
                    to_row = left_top_point[0] + mask_array.shape[0]
                    from_col = left_top_point[1]
                    to_col = left_top_point[1] + mask_array.shape[1]
                    mask_m[from_row:to_row, from_col:to_col] = mask_array
                    final_mask += mask_m

            with rio.open(
                (OUTPUT_PATH + file_name),
                "w",
                driver="GTiff",
                dtype=rio.uint8,
                count=1,
                width=mask_orig.width,
                height=mask_orig.height,
                transform=mask_orig.transform,
            ) as dst:
                # Before we colored polygon with white color for myself
                # checking,  to see masks in default windows "image viewer".
                # But for segmentation in torch we need to change 255 to 1.
                final_mask = np.where(final_mask < 1, final_mask, 1)
                dst.write(final_mask, indexes=1)

    # Merge all mask files to one big mask -----------------------------------
    # This part runnig long time near 10-15 minutes
    # TODO: optimize merge in future
    print("Start merge....")
    img_paths = glob.glob(os.path.join(OUTPUT_PATH, "*.tif"))
    img_paths.sort()
    tepmlate_to_full_mask = []
    for file in img_paths:
        template_mask = rio.open(file)
        tepmlate_to_full_mask.append(template_mask)
        temp_file, output = merge(tepmlate_to_full_mask)
        output_meta = template_mask.meta.copy()
        output_meta.update(
            {
                "driver": "GTiff",
                "height": temp_file.shape[1],
                "width": temp_file.shape[2],
                "transform": output,
            }
        )
        with rio.open(OUTPUT_FILE, "w", **output_meta) as m:
            m.write(temp_file)


if __name__ == "__main__":
    main()
