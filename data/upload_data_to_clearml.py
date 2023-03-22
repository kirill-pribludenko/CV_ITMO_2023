from clearml import Dataset, Task


task = Task.init(project_name='CV_MLOps_ITMO_2023',
                 task_name='upload final data')

path_to_folder_1_1 = './data/output/for_torchgeo_way/img_f'
path_to_folder_1_2 = './data/output/for_torchgeo_way/mask_f'

# For classic way maybe need change folder structure
# path_to_folder_2_1 = './data/output/for_classic_way/img'
# path_to_folder_2_2 = './data/output/for_classic_way/mask'

# Create a datasets with ClearML`s Dataset class
dataset_1 = Dataset.create(
    dataset_project="CV_MLOps_ITMO_2023",
    dataset_name="for_tocrhgeo"
)

# dataset_2 = Dataset.create(
#     dataset_project="CV_MLOps_ITMO_2023",
#     dataset_name="for_classic_way"
# )

# add files
dataset_1.add_files(path=path_to_folder_1_1, dataset_path='img_new_f')
dataset_1.add_files(path=path_to_folder_1_2, dataset_path='mask_new_f')
# dataset_2.add_files(path=path_to_folder_2_1)
# dataset_2.add_files(path=path_to_folder_2_2)

# Upload dataset to ClearML server (customizable)
dataset_1.upload()
# dataset_2.upload()

# commit dataset changes
dataset_1.finalize()
# dataset_2.finalize()
