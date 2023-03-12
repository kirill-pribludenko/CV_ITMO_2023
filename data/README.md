# Descrebing - How to create own custom dataset from Sentinel-2

**Note**: here only the creation of a dataset is described, the enrichment of classes through augmentation and other methods will be discussed elsewhere.


1. First need download data. You can choose different places, from which you can download data:
* [Copernicus Open Access Hub](https://scihub.copernicus.eu/dhus/#/home)
* [Earth Engine Data Catalog from Google](https://developers.google.com/earth-engine/datasets/)
* [Planetary Computer](https://planetarycomputer.microsoft.com/catalog)

For our project, we chose - **Copernicus Open Access Hub**. We have downloaded a part of the Tver region (Russia), in which the district of interest to us is located - Kashinsky. You can find the raw data that we used in the project at the [Google Drive link - raw data](https://drive.google.com/file/d/1cNIqu83s_tfcyiMj0WGQ9XLKQK_HzE76/view?usp=sharing)

2. It is necessary to perform some processing with raw data, because the format of their storage is atypical or not familiar to DS.
All manipulation you can find in file [processing_before_label.py](/data/processing_before_label.py). In short, this file:
* opening raw data
* do some simple manipulation with images
* creating RGB, RGBN and black template mask (N in RGBN- mean **NIR or near infrared** )
* splitting images to small pieces with dimension 512*512 (but you can set another)

3. Next, you need to make annotation. You can use any familiar tool for this, but we used [CVAT](https://www.cvat.ai/) for labeling. After you finish labeling, need to do export in CVAT format.

Our example of labeling.

<a href="/helpers/example_3.png"><img src="/helpers/example3.png" style="width: 500px; max-width: 100%; height: auto" title="Click for the larger version." /></a>

4. File `annotations.xml` need put to folder `/data/output`. If you have divided the annotation work into several people, you need to assemble everything into a single XML file by yourself (manually). Then run file [processing_after_label.py](/data/processing_after_label.py). In short, this file:
* opening XML file and create a dict, where **key** = name of file and **value** = temp dict with info about all polygons
* creating mask via fill all polygons white color for all small pieces (variant of dataset 1)
* creating one BIG mask for RGBN image (variant of dataset 2)

5. That all. Now you can use you own dataset for training, but before you need to split dataset to train & val:

* For **torchgeo** pls see file [torchgeo_split_ds.py](/data/torchgeo_split_ds.py) and run it, if you needed.
