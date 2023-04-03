from clearml import Dataset, Task


task = Task.init(project_name='CV_MLOps_ITMO_2023',
                 task_name='upload final data')

path_to_folder_1_1 = './data/processed/torchgeo/img'
path_to_folder_1_2 = './data/processed/torchgeo/mask'

path_to_folder_2_1 = './data/processed/classic/img'
path_to_folder_2_2 = './data/processed/classic/mask'

# Create a datasets with ClearML`s Dataset class
dataset_1 = Dataset.create(
    dataset_project="CV_MLOps_ITMO_2023",
    dataset_name="tocrhgeo"
)

dataset_2 = Dataset.create(
    dataset_project="CV_MLOps_ITMO_2023",
    dataset_name="classic"
)

# add files
dataset_1.add_files(path=path_to_folder_1_1, dataset_path='img_final')
dataset_1.add_files(path=path_to_folder_1_2, dataset_path='mask_final')
dataset_2.add_files(path=path_to_folder_2_1, dataset_path='img_final')
dataset_2.add_files(path=path_to_folder_2_2, dataset_path='img_final')

# Upload dataset to ClearML server (customizable)
dataset_1.upload()
dataset_2.upload()

# commit dataset changes
dataset_1.finalize()
dataset_2.finalize()
