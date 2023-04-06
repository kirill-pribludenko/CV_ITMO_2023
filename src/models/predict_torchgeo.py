import argparse
import os
import time
import warnings

import numpy as np
import rasterio as rio
import torch
from torchgeo.transforms import indices
from torchvision.models.segmentation import deeplabv3_resnet50
from torchvision.utils import draw_segmentation_masks

warnings.filterwarnings("ignore", category=rio.errors.NotGeoreferencedWarning)


def detect_heracleum():
    source, weights, img_size = opt.source, opt.weights, opt.img_size

    # Directories
    dir_for_save = './models/inference/results'

    # Load model
    model = deeplabv3_resnet50(weights=None, num_classes=2)
    backbone = model.get_submodule('backbone')

    conv = torch.nn.modules.conv.Conv2d(
        in_channels=5,
        out_channels=64,
        kernel_size=(7, 7),
        stride=(2, 2),
        padding=(3, 3),
        bias=False
    )

    backbone.register_module('conv1', conv)
    model.load_state_dict(torch.load(weights))
    model.eval()

    # Read input images
    files_paths = []
    for filename in os.listdir(source):
        filepath = os.path.join(source, filename)
        with rio.open(filepath) as img:
            # Check if the size of the images
            if img.width != img_size or img.height != img_size:
                img.close()
                print('Wrong size on input image')
            else:
                files_paths.append(filepath)

    # Add NDVI layer
    tfms = torch.nn.Sequential(indices.AppendNDVI(index_nir=3, index_red=0))

    # Run inference
    t0 = time.time()

    imgs_batch = torch.empty((len(files_paths), 5, img_size, img_size))

    for i, filepath in enumerate(files_paths):
        # open file
        img_a = rio.open(filepath).read()
        img_a = img_a.astype(np.float32)
        # from 0 - 255 to 0.0 - 1.0
        img_a /= 255.0
        img_t = torch.from_numpy(img_a)
        # add 1 dimension
        img_t = img_t[None, :, :, :]
        img_t = tfms(img_t)
        imgs_batch[i] = img_t

    pred = model(imgs_batch)['out']

    pred_time = round((time.time() - t0), 3)
    print(f'Predict Done. Time for {len(files_paths)} predict: {pred_time} sec')

    normalized_masks = torch.nn.functional.softmax(pred, dim=1)
    class_to_idx = {'no_heracleum': 0, 'heracleum': 1}
    boolean_masks = (normalized_masks.argmax(1) == class_to_idx['heracleum'])

    for i in range(len(files_paths)):
        img_t = (255*imgs_batch[i, :3]).type(torch.uint8)
        img_and_pred = draw_segmentation_masks(img_t,
                                               masks=boolean_masks[i],
                                               alpha=.6, colors='red')
        file_path = dir_for_save + f'/result_{i}.png'
        print(f'Saving file in {file_path}')
        img = rio.open(file_path, "w",
                       width=imgs_batch.shape[2],
                       height=imgs_batch.shape[2],
                       count=3,
                       dtype=np.uint8)
        img.write(img_and_pred)
        img.close()

    total_time = round((time.time() - t0), 3)
    print(f'Saving Done. Total time: {total_time} sec')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='test_torchgeo.pt',
                        help='write path to model.pt')
    parser.add_argument('--source', type=str, default='models/inference',
                        help='path to imgs')
    parser.add_argument('--img-size', type=int, default=256,
                        help='size of input img')
    opt = parser.parse_args()
    print(opt)

    with torch.no_grad():
        detect_heracleum()
