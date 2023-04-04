import gc
import os
from argparse import ArgumentParser
from time import perf_counter

import numpy as np
import rasterio as rio
import segmentation_models_pytorch as smp
import torch
import torch.optim as optim
from clearml import Dataset, Task
from lion_pytorch import Lion
from segmentation_models_pytorch.losses import DiceLoss, JaccardLoss
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.models.segmentation import deeplabv3_resnet50

# For ClearML
task = Task.init(project_name="CV_MLOps_ITMO_2023",
                 task_name="test_train_torch")
dataset_name = "classic"
dataset_project = "CV_MLOps_ITMO_2023"
dataset_path = Dataset.get(
    dataset_name=dataset_name, dataset_project=dataset_project
).get_local_copy()

parser = ArgumentParser()
parser.add_argument("--batch_size", type=int, default=2)
parser.add_argument("--loss_fns", type=str,
                    choices=["DiceLoss",
                             "JaccardLoss"],
                    default="DiceLoss")
parser.add_argument("--iou_thr", type=float, default=0.5)
parser.add_argument("--optimizer", type=str,
                    choices=["Adam",
                             "AdamW",
                             "Lion"],
                    default="Adam")
parser.add_argument("--model_lr", type=float, default=0.0001)
parser.add_argument("--model_epoch", type=int, default=2)

args = parser.parse_args()

config_dict = {
    "batch_size": args.batch_size,
    "loss_fns": args.loss_fns,
    "iou_thr": args.iou_thr,
    "optimizer": args.optimizer,
    "model_lr": args.model_lr,
    "model_epoch": args.model_epoch,
}

# Delete cuda cache
gc.collect()
torch.cuda.empty_cache()

tb_writer = SummaryWriter("./tb_logs")

# Check cuda and it is available
print(f"Cuda is available: {torch.cuda.is_available()}")
print(f"PyTorch version: {torch.__version__}")
print("*" * 10)
print(f"CUDNN version: {torch.backends.cudnn.version()}")
print(f"Available GPU devices: {torch.cuda.device_count()}")
print(f"Device Name: {torch.cuda.get_device_name()}")

# Creating Dataset, Sampler, Dataloader
print("Creating Dataset ...")
start = perf_counter()

i_train_path = dataset_path + "/img_final/train/"
i_val_path = dataset_path + "/img_final/val/"
m_train_path = dataset_path + "/mask_final/train/"
m_val_path = dataset_path + "/mask_final/val/"


class HeracleumDataset(torch.utils.data.Dataset):
    """Heracleum Dataset. Read images.
    Args:
        images_dir (str): path to images folder
        masks_dir (str): path to segmentation masks folder

    """

    def __init__(self, images_dir, masks_dir):
        self.image_paths = [
            os.path.join(images_dir, image_id)
            for image_id in sorted(os.listdir(images_dir))
        ]
        self.mask_paths = [
            os.path.join(masks_dir, image_id)
            for image_id in sorted(os.listdir(masks_dir))
        ]

    def __getitem__(self, i):
        # read images and masks
        image = rio.open(self.image_paths[i]).read().astype(np.float32)
        mask = rio.open(self.mask_paths[i]).read().astype(np.float32)

        return image, mask

    def __len__(self):
        # return length
        return len(self.image_paths)


train_dataset = HeracleumDataset(i_train_path, m_train_path)
val_dataset = HeracleumDataset(i_val_path, m_val_path)

train_loader = DataLoader(train_dataset,
                          batch_size=config_dict.get("batch_size", 2),
                          shuffle=True)
valid_loader = DataLoader(val_dataset,
                          batch_size=config_dict.get("batch_size", 2),
                          shuffle=False)

print("Define Losses & metrcis ...")

# Define loss fns and metric
if config_dict.get("loss_fns") == "DiceLoss":
    criterion = DiceLoss("multiclass")
elif config_dict.get("loss_fns") == "JaccardLoss":
    criterion = JaccardLoss("multiclass")
else:
    print("Wrong Loss Func name, check it")

metric = smp.utils.metrics.IoU(threshold=config_dict.get("iou_thr", 0.5))

print("Define Model ...")

# model, criterion, optimizer and scheduler
model = deeplabv3_resnet50(weights=None, num_classes=2, aux_loss=None)

if config_dict.get("optimizer") == "Adam":
    optimizer = optim.Adam(model.parameters(),
                           lr=config_dict.get("model_lr", 0.0001),
                           weight_decay=0.01)
elif config_dict.get("optimizer") == "AdamW":
    optimizer = optim.AdamW(model.parameters(),
                            lr=config_dict.get("model_lr", 0.0001),
                            weight_decay=0.01)
elif config_dict.get("optimizer") == "Lion":
    optimizer = Lion(model.parameters(),
                     lr=config_dict.get("model_lr", 0.0001),
                     weight_decay=0.01)
else:
    print("Wrong optimizer name, check it")

scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)


print("Start training ...")
# Training loop
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

for epoch in range(config_dict.get("model_epoch", 20)):
    model.train()
    train_loss = 0
    train_iou = 0
    for batch in train_loader:
        images = batch[0]
        labels = batch[1].long()

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)["out"]
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        preds = torch.argmax(outputs, dim=1)
        train_iou += metric(labels, preds)

    train_loss /= len(train_loader)
    train_iou /= len(train_loader)

    model.eval()
    val_loss = 0
    val_iou = 0
    with torch.no_grad():
        for batch in valid_loader:
            images = batch[0]
            labels = batch[1].long()

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)["out"]
            loss = criterion(outputs, labels)

            val_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            val_iou += metric(labels, preds)

    val_loss /= len(valid_loader)
    val_iou /= len(valid_loader)

    scheduler.step()

    print(f"Epoch {epoch + 1}/{config_dict.get('model_epoch', 20)}")
    print(f"Train Loss: {train_loss:.4f}, Train IoU: {train_iou:.4f}")
    print(f"Val Loss: {val_loss:.4f}, Val IoU: {val_iou:.4f}")
    tb_writer.add_scalars("Loss",
                          {"train": train_loss, "val": val_loss}, epoch + 1)
    tb_writer.add_scalars("IoU",
                          {"train": train_iou, "val": val_iou}, epoch + 1)

print("Save model ...")
torch.save(model.state_dict(), "./test_torch.pt")
stop = perf_counter()
timer = (stop - start) / 60
print(f"Total time: {timer:.2f} min")
