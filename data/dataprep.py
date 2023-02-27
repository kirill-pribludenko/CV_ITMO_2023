import rasterio
from rasterio import plot
from pathlib import Path
import matplotlib.pyplot as plt


BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / 'data'
IMG_DIR = DATA_DIR / 'raw' / 'S2A_MSIL2A_20220703T084611_N0400_R107_T37VCD_20220703T134122' / 'S2A_MSIL2A_20220703T084611_N0400_R107_T37VCD_20220703T134122.SAFE' / 'GRANULE' / 'L2A_T37VCD_A036712_20220703T085107' / 'IMG_DATA' / 'R10m'
print(IMG_DIR)
# TODO нужно сделать обработку, скорее всего нарезку сырых данных