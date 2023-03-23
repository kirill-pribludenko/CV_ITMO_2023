import gc
from argparse import ArgumentParser
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
import torch
import torch.optim as optim
from clearml import Dataset, Task
from segmentation_models_pytorch.losses import DiceLoss, FocalLoss, JaccardLoss
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchgeo.datasets import RasterDataset, stack_samples
from torchgeo.samplers import GridGeoSampler, RandomGeoSampler, Units
from torchgeo.transforms import indices
from torchvision.models.segmentation import deeplabv3_resnet50

# For ClearML
task = Task.init(project_name="CV_MLOps_ITMO_2023", task_name="test2_train_torchgeo")
dataset_name = "for_tocrhgeo"
dataset_project = "CV_MLOps_ITMO_2023"
dataset_path = Dataset.get(
    dataset_name=dataset_name, dataset_project=dataset_project
).get_local_copy()

parser = ArgumentParser()
parser.add_argument("--img_size", type=int, default=256)
parser.add_argument("--train_samp_len", type=int, default=640)
parser.add_argument("--val_samp_len", type=int, default=320)
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument('--loss_fns', type=str,
                    choices=['DiceLoss',
                             'FocalLoss',
                             'JaccardLoss'],
                    default='DiceLoss')
parser.add_argument("--iou_thr", type=float, default=0.5)
parser.add_argument("--model_lr", type=float, default=0.0001)
parser.add_argument("--model_epoch", type=int, default=2)

args = parser.parse_args()

config_dict = {
    "img_size": args.img_size,
    "train_samp_len": args.train_samp_len,
    "val_samp_len": args.val_samp_len,
    "batch_size": args.batch_size,
    'loss_fns': args.loss_fns,
    "iou_thr": args.iou_thr,
    "model_lr": args.model_lr,
    "model_epoch": args.model_epoch,
}

# Delete cuda cache
gc.collect()
torch.cuda.empty_cache()

tb_writer = SummaryWriter("./tb_logs")
# root = Path("./data/output/for_torchgeo_way")
# assert root.exists()

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


def scale(item: dict):
    item["image"] = (item["image"] - torch.min(item["image"])) / (
        torch.max(item["image"]) - torch.min(item["image"])
    )
    return item


train_imgs = RasterDataset(
    root=(dataset_path + "/img_new_f/train/"), crs="epsg:32637", res=10, transforms=scale
)
train_msks = RasterDataset(
    root=(dataset_path + "/mask_new_f/train/"), crs="epsg:32637", res=10
)

valid_imgs = RasterDataset(
    root=(dataset_path + "/img_new_f/val/"), crs="epsg:32637", res=10, transforms=scale
)
valid_msks = RasterDataset(
    root=(dataset_path + "/mask_new_f/val/"), crs="epsg:32637", res=10
)

train_msks.is_image = False
valid_msks.is_image = False

train_dset = train_imgs & train_msks
valid_dset = valid_imgs & valid_msks

train_sampler = RandomGeoSampler(
    train_imgs,
    size=config_dict.get("img_size", 256),
    length=config_dict.get("train_samp_len", 640),
    units=Units.PIXELS,
)
valid_sampler = RandomGeoSampler(
    valid_imgs,
    size=config_dict.get("img_size", 256),
    length=config_dict.get("val_samp_len", 320),
    units=Units.PIXELS,
)

# train_sampler = GridGeoSampler(train_imgs, size=256, stride=250, units=Units.PIXELS)
# valid_sampler = GridGeoSampler(valid_imgs, size=256, stride=250, units=Units.PIXELS)

train_loader = DataLoader(
    train_dset,
    sampler=train_sampler,
    batch_size=config_dict.get("batch_size", 8),
    #   batch_size=8,
    collate_fn=stack_samples,
)
valid_loader = DataLoader(
    valid_dset,
    sampler=valid_sampler,
    batch_size=config_dict.get("batch_size", 8),
    #   batch_size=8,
    collate_fn=stack_samples,
)

# Add new layer - NDVI
tfms = torch.nn.Sequential(
    indices.AppendNDVI(index_nir=3, index_red=0),
)

print("Define Losses & metrcis ...")

# Define loss fns and metric
if config_dict.get("loss_fns") == 'DiceLoss':
    criterion = DiceLoss("multiclass")
elif config_dict.get("loss_fns") == 'FocalLoss':
    criterion = FocalLoss('multiclass')
elif config_dict.get("loss_fns") == 'JaccardLoss':
    criterion = JaccardLoss('multiclass')
else:
    print('Wrong Loss Func name, check it')

metric = smp.utils.metrics.IoU(threshold=config_dict.get("iou_thr", 0.5))

print("Define Model ...")

# model, criterion, optimizer and scheduler
model = deeplabv3_resnet50(weights=None, num_classes=2, aux_loss=None)
backbone = model.get_submodule("backbone")
conv = torch.nn.modules.conv.Conv2d(
    in_channels=5,
    out_channels=64,
    kernel_size=(7, 7),
    stride=(2, 2),
    padding=(3, 3),
    bias=False,
)
backbone.register_module("conv1", conv)

optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.01)
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
        images = batch["image"]
        labels = batch["mask"]

        images = tfms(images)

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
            images = batch["image"]
            labels = batch["mask"]

            images = tfms(images)

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
    tb_writer.add_scalars("Loss", {"train": train_loss, "val": val_loss}, epoch + 1)
    tb_writer.add_scalars("IoU", {"train": train_iou, "val": val_iou}, epoch + 1)

print("Save model ...")
torch.save(model.state_dict(), "./test.pt")
stop = perf_counter()
timer = (stop - start) / 60
print(f"Total time: {timer:.2f} min")
