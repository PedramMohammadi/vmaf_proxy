#!/usr/bin/env python3

import argparse
import json
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import DataLoader as TorchDataLoader
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from scipy.stats import pearsonr, spearmanr
from .vmaf_data_loader import VMAFDataset
from .model import VMAFNet
import random
from google.cloud import storage
import torch.optim as optim
import torch.multiprocessing as mp
import os
from torch.utils.data._utils.collate import default_collate
from torch.cuda.amp import autocast, GradScaler

# ------------------------------
# Adding random seed to guarantee reproducibility
# ------------------------------
seed = 42  # Or make it an argparse arg
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True

# ------------------------------
# Helpers
# ------------------------------
def collate_skip_none(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    return default_collate(batch)

def worker_init_fn(worker_id):
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training a VMAF proxy model using 3D CNN architecture.")
    parser.add_argument("--csv_file", type=str, required=True, help="Path to the CSV file containing frame paths and VMAF scores.")
    parser.add_argument("--root_dir", type=str, required=True, help="Root directory where frames are stored.")
    parser.add_argument("--output_dir", type=str, default="output", help="Directory to save model checkpoints, metrics, and plots.")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training and validation.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for the optimizer.")
    parser.add_argument("--norm_groups", type=int, default=16, help="Groups per conv block. Ideally a divisor of the layer's channel count.")
    parser.add_argument("--kernel_size", type=int, default=3, help="3D convolution kernel size for. Must be an odd number")
    parser.add_argument("--num_conv_layers", type=int, default=7, help="Number of convolution blocks")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay parameter for the optimizer")
    parser.add_argument("--reduction", type=int, default=16, help="Squeeze-and-Excitation reduction ratio.")
    parser.add_argument("--early_stop_patience", type=int, default=10, help="Number of maximum consecutive epochs without validation loss improvement to stop training.")
    parser.add_argument("--use_plateau_scheduler", action="store_true", help="Modify the learning rate when learning plateaus.")
    parser.add_argument("--width", type=float, default=0.75, help="Width multiplier for channel counts in the model.")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout probability in convolutional layers.")
    parser.add_argument("--activation", type=str, default="leaky_relu", choices=["relu", "leaky_relu"], help="Activation function in the model ('relu' or 'leaky_relu'; default: 'relu').")
    parser.add_argument("--crop_size", type=int, default=128, help="Size of square crop patches from frames.")
    parser.add_argument("--n_frames", type=int, default=3, help="Number of consecutive frames per sample.")
    parser.add_argument("--early_stop", action="store_true", help="Enable early stopping based on validation loss.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device for training.")
    parser.add_argument("--num_workers", type=int, default=2, help="Number of workers for data loading.")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint file to resume training from.")
    args = parser.parse_args()

    if args.kernel_size % 2 == 0:
        raise ValueError(f"--kernel_size must be odd (got {args.kernel_size})")
    if args.n_frames % 2 == 0:
        raise ValueError(f"--n_frames must be odd (got {args.n_frames})")

    mp.set_start_method('spawn', force=True)

    train_dataset = VMAFDataset(
        csv_file=args.csv_file,
        root_dir=args.root_dir,
        crop_size=args.crop_size,
        fixed_crop=False,
        n_frames=args.n_frames,
        split="Training"
    )
    val_dataset = VMAFDataset(
        csv_file=args.csv_file,
        root_dir=args.root_dir,
        crop_size=args.crop_size,
        fixed_crop=True,
        n_frames=args.n_frames,
        split="Validation"
    )

    train_loader = TorchDataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, prefetch_factor=4,
        pin_memory=args.device.startswith("cuda"),
        persistent_workers=(args.num_workers > 0),
        drop_last=False,
        worker_init_fn=worker_init_fn, collate_fn=collate_skip_none
    )
    val_loader = TorchDataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, prefetch_factor=4,
        pin_memory=args.device.startswith("cuda"),
        persistent_workers=(args.num_workers > 0),
        drop_last=False,
        worker_init_fn=worker_init_fn, collate_fn=collate_skip_none
    )

    model = VMAFNet(
            width=args.width,
            dropout=args.dropout,
            activation=args.activation,
            norm_groups=args.norm_groups,
            kernel_size=args.kernel_size,
            reduction=args.reduction,
            num_conv_layers=args.num_conv_layers
        ).to(args.device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    scaler = GradScaler()

    if args.use_plateau_scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=2, min_lr=1e-6, verbose=True
        )
    else:
        scheduler = None
    start_epoch = 0
    best_val_loss = float('inf')
    early_stop_counter = 0
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.resume:
        resume_path = args.resume

        # If a directory was passed, try the file we actually save
        if os.path.isdir(resume_path):
            resume_path = str(Path(resume_path) / "model.pth")
        
        # If it's a GCS URI, download it locally first
        if isinstance(resume_path, str) and resume_path.startswith("gs://"):
            if storage is None:
                raise RuntimeError("google-cloud-storage is required to resume from gs://")
            uri = resume_path[len("gs://"):]
            bucket_name, _, blob_name = uri.partition("/")
            client = storage.Client()
            bucket = client.bucket(bucket_name)
            blob = bucket.blob(blob_name)

            local_dir = Path(args.output_dir) if args.output_dir else Path("/tmp/checkpoints")
            local_dir.mkdir(parents=True, exist_ok=True)
            local_resume = str(local_dir / "resume_checkpoint.pth")
            blob.download_to_filename(local_resume)
            print(f"Downloaded resume checkpoint from {args.resume} to {local_resume}")
            resume_path = local_resume
        
        if os.path.isfile(resume_path):
            print(f"Resuming from {resume_path}")
            checkpoint = torch.load(resume_path, map_location=args.device)
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            if args.use_plateau_scheduler and checkpoint.get('scheduler') is not None:
                scheduler.load_state_dict(checkpoint['scheduler'])
            start_epoch = checkpoint.get('epoch', -1) + 1
            best_val_loss = checkpoint.get('best_val_loss', float('inf'))

            #if you want to override LR after resume:
            for pg in optimizer.param_groups:
                pg['lr'] = args.lr
        else:
            print(f"WARNING: --resume specified but file not found: {resume_path}")

    train_losses, val_losses, val_mae, val_plcc, val_srcc = [], [], [], [], []
    
    for epoch in range(start_epoch, args.epochs):
        model.train()
        running_loss, num_train_batches = 0.0, 0.0

        for batch in tqdm(train_loader, disable=True):
            
            if batch is None:
                continue
            
            ref, dist, target = batch
            ref, dist, target = ref.to(args.device), dist.to(args.device), target.to(args.device)

            optimizer.zero_grad()
            with autocast():
                prediction = model(ref, dist).view(-1)
                prediction = prediction.squeeze()
                loss = criterion(prediction, target.view(-1))

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            num_train_batches += 1

        avg_train_loss = running_loss / max(1, num_train_batches)
        train_losses.append(avg_train_loss)

        # Validation
        model.eval()
        val_loss, num_val_batches = 0.0, 0.0
        preds, targets = [], []

        with torch.no_grad():
            for batch in tqdm(val_loader, disable=True):

                if batch is None:
                    continue
                
                ref, dist, target = batch
                ref, dist, target = ref.to(args.device), dist.to(args.device), target.to(args.device)

                with autocast():
                    prediction = model(ref, dist).view(-1)
                    prediction = prediction.squeeze()
                    loss = criterion(prediction, target.view(-1))

                val_loss += loss.item()
                num_val_batches += 1
                preds.append((prediction * 100).detach().cpu())
                targets.append((target * 100).detach().cpu())

        avg_val_loss = val_loss / max(1, num_val_batches)
        val_losses.append(avg_val_loss)
        mae = torch.abs(torch.cat(preds) - torch.cat(targets)).mean().item()
        val_mae.append(mae)
        plcc = pearsonr(torch.cat(preds).numpy(), torch.cat(targets).numpy())[0]
        srcc = spearmanr(torch.cat(preds).numpy(), torch.cat(targets).numpy())[0]
        val_plcc.append(plcc)
        val_srcc.append(srcc)

        print(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}, Val Loss={avg_val_loss:.4f}, MAE={mae:.4f}, PLCC={plcc:.3f}, SRCC={srcc:.3f}")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict() if scheduler else None,
                'epoch': epoch,
                'best_val_loss': best_val_loss
            }, output_dir / "model.pth")

            client = storage.Client()
            bucket = client.bucket('vmaf_proxy_training_checkpoints')
            blob = bucket.blob('checkpoints/model.pth')
            blob.upload_from_filename(output_dir / "model.pth")
            print("Checkpoint uploaded to GCS")

            early_stop_counter = 0
        else:
            early_stop_counter += 1

        if scheduler:
            scheduler.step(avg_val_loss)

        if args.early_stop and early_stop_counter >= args.early_stop_patience:
            print("Early stopping triggered.")
            break

    with open(output_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)
    
    with open(output_dir / "metrics.json", "w") as f:
        json.dump({
            "train_loss": train_losses,
            "val_loss": val_losses,
            "val_mae": val_mae,
            "val_plcc": val_plcc,
            "val_srcc": val_srcc
        }, f, indent=2)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Train")
    plt.plot(val_losses, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training & Validation Loss")
    plt.subplot(1, 2, 2)
    plt.plot(val_plcc, label="PLCC")
    plt.plot(val_srcc, label="SRCC")
    plt.xlabel("Epoch")
    plt.ylabel("Correlation")
    plt.legend()
    plt.title("Validation PLCC & SRCC")
    plt.tight_layout()
    plt.savefig(output_dir / "metrics_curve.png")
    plt.close()

    client = storage.Client()
    bucket = client.bucket('vmaf_proxy_training_checkpoints')
    for file in ["config.json", "metrics.json", "metrics_curve.png"]:
        blob = bucket.blob(f'outputs/{file}')
        blob.upload_from_filename(output_dir / file)
        print(f"{file} uploaded to GCS")