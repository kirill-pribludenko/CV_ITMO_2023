import glob
import os
import xml.etree.ElementTree as ET

import cv2
import numpy as np
import rasterio as rio
from rasterio.merge import merge

LABEL_FILE_NAME = "./data/output/annotations.xml"
INPUT_MASKS_DIR = "./data/output/for_classic_way/mask_template"
OUTPUT_PATH = "./data/output/for_classic_way/mask"
OUTPUT_FILE = "./data/output/for_torchgeo_way/mask.tif"


def main():
    # Reading xml file with coordinates of polygon mask
    # for each file and generate dict {'name_of_file': '[array_points]'}
    tree = ET.parse(LABEL_FILE_NAME)
    root = tree.getroot()
    dict_masks = {}
    for i in range(2, len(root)):
        text_coords = root[i][0].attrib["points"].replace(";", ",")
        array_points = np.array(text_coords.split(",")).astype(float)
        array_points = array_points.reshape(-1, 2).astype(int)

        # changing name from img_* to mask_* in dict
        correct_name = root[i].attrib["name"]
        correct_name = "mask" + correct_name[3:]
        dict_masks[correct_name] = array_points

    # Opening template of masks and fill polygons with
    # white color for each file in dict
    for file_name in dict_masks.keys():
        mask_orig = rio.open(INPUT_MASKS_DIR + file_name)
        img_array = mask_orig.read(1)
        mask = cv2.fillPoly(
            img_array, pts=[dict_masks[file_name]], color=(255, 255, 255)
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
        mask_orig.close()

    # Merge all mask files to one big mask
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
