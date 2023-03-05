import glob
import os
import xml.etree.ElementTree as ET

import cv2
import numpy as np
import rasterio as rio
from rasterio.merge import merge

LABEL_FILE_NAME = "./data/output/annotations.xml"
INPUT_MASKS_DIR = "./data/output/for_classic_way/mask_template/"
OUTPUT_PATH = "./data/output/for_classic_way/mask/"
OUTPUT_FILE = "./data/output/for_torchgeo_way/mask.tif"


def main():
    # Reading xml file with coordinates of polygon mask
    # for each file and generate dict with coord of all masks
    # Structure of dict
    # {'name_of_file_N':
    #   {'polygon_N': '[array_points]'}
    # } - if polygons exist
    # {'name_of_file_N': False
    # } - if polygons not exist

    tree = ET.parse(LABEL_FILE_NAME)
    root = tree.getroot()
    dict_masks = {}

    # Opening XML file and create dict ----------------------------------------
    for i in range(2, len(root)):
        # changing name from img_* to mask_* for dict
        correct_name = root[i].attrib["name"]
        correct_name = "mask" + correct_name[3:]

        if len(root[i]) > 0:
            dict_temp = {}
            for j in range(len(root[i])):
                text_coords = root[i][j].attrib["points"].replace(";", ",")
                array_points = np.array(text_coords.split(",")).astype(float)
                array_points = array_points.reshape(-1, 2).astype(int)
                dict_temp[f"polygon_{j}"] = array_points
            dict_masks[correct_name] = dict_temp
        else:
            dict_masks[correct_name] = False

    # Fill all polygons in templates masks ------------------------------------
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
            for polygon_array in dict_masks[file_name].values():
                mask = cv2.fillPoly(
                    img_array, pts=[polygon_array], color=(255, 255, 255)
                )
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
                dst.write(mask, indexes=1)
        # closing file
        mask_orig.close()

    # Merge all mask files to one big mask ----------------------------------
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
