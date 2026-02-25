# Copyright 2025 Pierre-Yves Nicolas

# This program is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the Free
# Software Foundation, either version 3 of the License, or (at your option)
# any later version.
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
# FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for
# more details.
# You should have received a copy of the GNU General Public License along with
# this program. If not, see <https://www.gnu.org/licenses/>.

import os
import cv2
import glob
import zipfile
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms.functional as TF
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
from urllib.request import urlretrieve
import shutil

DATASET_VERSION = 'v2.1'
DATASET_ZIP_URL = f'https://github.com/pynicolas/fairscan-dataset/releases/download/{DATASET_VERSION}/fairscan-dataset-{DATASET_VERSION}.zip'

BUILD_DIR = "build"
MODEL_DIR = BUILD_DIR + "/model"
MODEL_FILE_PATH = MODEL_DIR + "/fairscan-segmentation-model.pt"
TFLITE_MODEL_FILE_PATH = MODEL_DIR + "/fairscan-segmentation-model.tflite"
DATASET_ZIP_PATH = BUILD_DIR + "/dataset.zip"
DATASET_PARENT_DIR = BUILD_DIR + "/dataset"
DATASET_DIR = DATASET_PARENT_DIR + "/fairscan-dataset"
NB_EPOCHS = 1

if os.path.isdir(BUILD_DIR):
    shutil.rmtree(BUILD_DIR)
os.makedirs(BUILD_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
shutil.copy("LICENSE", MODEL_DIR + "/LICENSE.txt")

# Dataset

print('Download and extract dataset...')
urlretrieve(DATASET_ZIP_URL, DATASET_ZIP_PATH)
with zipfile.ZipFile(DATASET_ZIP_PATH, 'r') as zip_ref:
    zip_ref.extractall(DATASET_PARENT_DIR)

class DocumentSegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_paths = sorted(glob.glob(os.path.join(image_dir, "*.jpg")))
        self.mask_paths = sorted(glob.glob(os.path.join(mask_dir, "*.png")))
        self.transform = transform

    def __getitem__(self, idx):
        image = cv2.imread(self.image_paths[idx])  # BGR
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)

        mask = (mask > 127).astype('float32')

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask'].unsqueeze(0)  # (1, H, W)
        else:
            image = TF.to_tensor(image)
            mask = TF.to_tensor(mask)

        return image, mask

    def __len__(self):
        return len(self.image_paths)


# Data loaders

print('Set up data loaders...')

shared_transform = A.Compose([
    A.Resize(256, 256),
    A.Normalize(),
    ToTensorV2(),
])

train_transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    shared_transform,
])

train_dataset = DocumentSegmentationDataset(
    image_dir=os.path.join(DATASET_DIR, "train/images"),
    mask_dir=os.path.join(DATASET_DIR, "train/masks"),
    transform=train_transform
)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, drop_last=True)

val_dataset = DocumentSegmentationDataset(
    image_dir=os.path.join(DATASET_DIR, "val/images"),
    mask_dir=os.path.join(DATASET_DIR, "val/masks"),
    transform=shared_transform
)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)


# Training

def dice_score_continuous(pred, target, smooth=1e-6):
    # pred and target: [B, 1, H, W] or [B, H, W]
    pred_flat = pred.contiguous().view(-1)
    target_flat = target.contiguous().view(-1)
    intersection = (pred_flat * target_flat).sum()
    return (2. * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)

def dice_score_discrete(pred, target, smooth=1e-6):
    pred_bin = (pred > 0.5).float()
    return dice_score_continuous(pred_bin, target, smooth)

dice_loss = smp.losses.DiceLoss(mode='binary')
bce_loss = nn.BCEWithLogitsLoss()

def loss_fn(pred, target):
    return dice_loss(pred, target) + bce_loss(pred, target)

def evaluate_encoder(encoder_name, model_save_path, device=torch.device('cpu')):
    model = smp.DeepLabV3Plus(
        encoder_name=encoder_name,
        encoder_weights="imagenet",
        in_channels=3,
        classes=1
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    best_dice = -1
    best_state = None

    for epoch in range(NB_EPOCHS):
        start = time.time()

        model.train()
        total_loss = 0
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            preds = model(images)
            loss = loss_fn(preds, masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_train_loss = total_loss / len(train_loader)

        model.eval()
        val_loss = 0.0
        dice_cont_sum = 0.0
        dice_disc_sum = 0.0

        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)

                outputs = model(images)
                probs = torch.sigmoid(outputs)

                loss = loss_fn(probs, masks)
                val_loss += loss.item()

                dice_cont_sum += dice_score_continuous(probs, masks).item()
                dice_disc_sum += dice_score_discrete(probs, masks).item()

        val_loss /= len(val_loader)
        dice_cont_mean = dice_cont_sum / len(val_loader)
        dice_disc_mean = dice_disc_sum / len(val_loader)
        end = time.time()

        print(f"- Epoch {epoch + 1}/{NB_EPOCHS}: train_loss={avg_train_loss:.4f} | Val Loss: {val_loss:.4f}" +
              f" | Dice (cont): {dice_cont_mean:.4f} | Dice (disc): {dice_disc_mean:.4f} | {end - start:.1f} seconds")

        if dice_disc_mean > best_dice:
            best_dice = dice_disc_mean
            best_state = model.state_dict()

    # Save best model temporarily to measure size
    torch.save(best_state, model_save_path)
    print(f"Wrote {MODEL_FILE_PATH}")
    model_size_mb = os.path.getsize(model_save_path) / 1e6

    # Inference time test
    dummy_input = torch.randn(1, 3, 256, 256).to(device)
    model.eval()
    model.load_state_dict(best_state)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start_time = time.time()
    for _ in range(10):
        _ = model(dummy_input)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    inference_time_ms = (time.time() - start_time) * 100 / 10

    return {
        "encoder": encoder_name,
        "dice": round(best_dice, 4),
        "size_mb": round(model_size_mb, 2),
        "inference_ms": round(inference_time_ms, 2)
    }


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

encoder = "mobilenet_v2"
# Other possible encoders:
# "timm-tf_efficientnet_lite1"
# "timm-efficientnet-b0"
# "mobilenet_v2"
# "mobileone_s0"

print(f"Training {encoder}...")
result = evaluate_encoder(encoder, model_save_path=MODEL_FILE_PATH, device=device)
print(result)

import json
with open(os.path.join(MODEL_DIR, "metrics.json"), "w") as f:
    json.dump(result, f, indent=2)

# Convert to TFLite

import litert_torch
from litert_torch.generative.quantize import quant_recipes

model = smp.DeepLabV3Plus(
    encoder_name=encoder,
    encoder_weights=None,
    in_channels=3,
    classes=1,
)
model.load_state_dict(torch.load(MODEL_FILE_PATH, map_location="cpu"))
model.eval()

class NHWCWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input: (N, H, W, C) → PyTorch → (N, C, H, W)
        x = x.permute(0, 3, 1, 2)
        x = self.model(x)

        # (N, C, H, W) → (N, H, W, C)
        if x.ndim == 4:
            x = x.permute(0, 2, 3, 1)
        return x
wrapped_model = NHWCWrapper(model)
wrapped_model.eval()

# 1. sample_args to trace graph
sample_input = torch.randn(1, 256, 256, 3)  # NHWC
sample_args = (sample_input,)

# 2. representative_dataset to calibrate quantization
quantization_dataset = DocumentSegmentationDataset(
    image_dir=os.path.join(DATASET_DIR, "val/images"),
    mask_dir=os.path.join(DATASET_DIR, "val/masks"),
    transform=shared_transform
)
quantization_loader = DataLoader(quantization_dataset, batch_size=4, shuffle=False)
def representative_dataset():
    for images, _ in quantization_loader:
        for i in range(images.size(0)):
            img = images[i].permute(1, 2, 0).unsqueeze(0)  # C,H,W → H,W,C → NHWC
            img = img.to(torch.float32)
            yield (img,)

# 3. quant_config
quant_config = quant_recipes.full_dynamic_recipe()

# 4. Conversion
edge_model_quantized = litert_torch.convert(
    wrapped_model,
    sample_args=sample_args,
    sample_kwargs=None,
    quant_config=quant_config,
    _ai_edge_converter_flags={"representative_dataset": representative_dataset}
)
edge_model_quantized.export(TFLITE_MODEL_FILE_PATH)
print(f"Wrote TFLite model: {TFLITE_MODEL_FILE_PATH}")
