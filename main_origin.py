#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
U-Net を用いたセグメンテーション（一発実行スクリプト, seed対応）

追加点:
- --seed で乱数シード固定（Python / NumPy / PyTorch, DataLoader workers, cuDNN）
- DataLoader に worker_init_fn / generator を設定
"""

import os
import argparse
from typing import Tuple
from datetime import datetime
import random

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

import torchvision.transforms as T
from torchvision.datasets import OxfordIIITPet
from torchvision.utils import save_image

from tqdm import tqdm  # 進捗バー
import matplotlib.pyplot as plt  # 学習曲線

import zipfile
import io
import requests  # DUTS 自動ダウンロード

try:
    import segmentation_models_pytorch as smp  # U-Net 実装
except Exception as e:
    raise ImportError(
        "segmentation_models_pytorch is required. Install via 'pip install segmentation-models-pytorch timm'"
    ) from e


# ==============================
# シード固定ユーティリティ
# ==============================
def seed_everything(seed: int):
    # 乱数の再現性を確保する設定
    os.environ["PYTHONHASHSEED"] = str(seed)          # Pythonのハッシュのランダム性を固定
    random.seed(seed)                                  # Python標準乱数
    np.random.seed(seed)                               # NumPy乱数
    torch.manual_seed(seed)                            # PyTorch CPU
    torch.cuda.manual_seed_all(seed)                   # PyTorch CUDA 全GPU
    # cuDNN を決定論的に（速度低下の可能性あり）
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # 可能なら決定論的アルゴリズムのみ使用（未対応の演算があると例外の可能性）
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass

def seed_worker(worker_id: int):
    # DataLoader の各 worker プロセス内で NumPy / random を初期化
    worker_seed = (torch.initial_seed() + worker_id) % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# ------------------------------
# データセット存在確認 & ダウンロードユーティリティ
# ------------------------------

def oxford_pet_exists(root: str) -> bool:
    base = os.path.join(root, "oxford-iiit-pet")
    images = os.path.join(base, "images")
    ann = os.path.join(base, "annotations")
    return os.path.isdir(images) and os.path.isdir(ann)

def duts_exists(root: str) -> bool:
    tr_img = os.path.join(root, "DUTS-TR", "DUTS-TR-Image")
    tr_msk = os.path.join(root, "DUTS-TR", "DUTS-TR-Mask")
    te_img = os.path.join(root, "DUTS-TE", "DUTS-TE-Image")
    te_msk = os.path.join(root, "DUTS-TE", "DUTS-TE-Mask")
    return all(os.path.isdir(p) for p in [tr_img, tr_msk, te_img, te_msk])

def download_and_extract_zip(url: str, dest_root: str):
    os.makedirs(dest_root, exist_ok=True)
    print(f"[Info] Downloading: {url}")
    r = requests.get(url, stream=True, timeout=60)
    r.raise_for_status()
    z = zipfile.ZipFile(io.BytesIO(r.content))
    z.extractall(dest_root)
    print("[Info] Extracted")

def prepare_duts(root: str):
    if duts_exists(root):
        print("[Info] DUTS already exists. Skipping download.")
        return
    train_url = "http://saliencydetection.net/duts/download/DUTS-TR.zip"
    test_url  = "http://saliencydetection.net/duts/download/DUTS-TE.zip"
    download_and_extract_zip(train_url, root)
    download_and_extract_zip(test_url, root)
    print("[Info] DUTS prepared.")


# ---- Custom Binary Segmentation Dataset ----
class CustomBinarySegDataset(torch.utils.data.Dataset):
    def __init__(self, img_dir, mask_dir, image_size=256,
                 img_exts=(".jpg", ".jpeg", ".png", ".bmp"),
                 threshold=127):
        super().__init__()
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.size = image_size
        self.threshold = threshold
        names = []
        for f in os.listdir(img_dir):
            if f.lower().endswith(img_exts) and not f.startswith('.'):
                names.append(os.path.splitext(f)[0])
        self.names = sorted(names)

        self.img_tf = T.Compose([
            T.Resize((image_size, image_size), interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.names)

    def _open_image(self, stem):
        for ext in [".jpg", ".jpeg", ".png", ".bmp"]:
            p = os.path.join(self.img_dir, stem + ext)
            if os.path.isfile(p):
                return Image.open(p).convert("RGB")
        raise FileNotFoundError(f"Image not found for {stem}")

    def _open_mask(self, stem):
        for ext in [".png", ".jpg", ".bmp"]:
            p = os.path.join(self.mask_dir, stem + ext)
            if os.path.isfile(p):
                return Image.open(p).convert("L")
        raise FileNotFoundError(f"Mask not found for {stem}")

    def __getitem__(self, idx):
        stem = self.names[idx]
        img = self._open_image(stem)
        mask = self._open_mask(stem)

        img = self.img_tf(img)
        mask = mask.resize((self.size, self.size), resample=Image.NEAREST)

        m_np = (np.array(mask, dtype=np.uint8) > self.threshold).astype(np.float32)
        m = torch.from_numpy(m_np).unsqueeze(0)  # (1, H, W)
        return img, m


class OxfordPetSegBinary(Dataset):
    def __init__(self, root: str, split: str = "trainval", image_size: int = 256, download: bool = True):
        super().__init__()
        self.base = OxfordIIITPet(root=root, split=split, download=download, target_types=("segmentation",))
        self.img_tf = T.Compose([
            T.Resize((image_size, image_size), interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.mask_tf = T.Compose([
            T.Resize((image_size, image_size), interpolation=T.InterpolationMode.NEAREST),
        ])

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img, target = self.base[idx]
        img = self.img_tf(img)
        mask_pil: Image.Image = self.mask_tf(target)
        mask_np = np.array(mask_pil, dtype=np.uint8)
        bin_np = np.where(mask_np == 3, 0, 1).astype(np.float32)
        mask = torch.from_numpy(bin_np).unsqueeze(0)
        return img, mask


class DUTSSaliencyDataset(Dataset):
    def __init__(self, root: str, split: str = "train", image_size: int = 256):
        super().__init__()
        base = "DUTS-TR" if split == "train" else "DUTS-TE"
        self.img_dir = os.path.join(root, base, f"{base}-Image")
        self.msk_dir = os.path.join(root, base, f"{base}-Mask")
        self.names = sorted([os.path.splitext(f)[0] for f in os.listdir(self.img_dir) if not f.startswith('.')])
        self.img_tf = T.Compose([
            T.Resize((image_size, image_size), interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.size = image_size

    def __len__(self):
        return len(self.names)

    def _open_mask(self, stem: str) -> Image.Image:
        for ext in [".png", ".jpg", ".bmp"]:
            p = os.path.join(self.msk_dir, stem + ext)
            if os.path.isfile(p):
                return Image.open(p).convert("L")
        raise FileNotFoundError(f"Mask not found for {stem}")

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        stem = self.names[idx]
        img_path = None
        for ext in [".jpg", ".png", ".bmp"]:
            p = os.path.join(self.img_dir, stem + ext)
            if os.path.isfile(p):
                img_path = p
                break
        if img_path is None:
            raise FileNotFoundError(f"Image not found for {stem}")
        img = Image.open(img_path).convert("RGB")
        msk = self._open_mask(stem)
        img = self.img_tf(img)
        msk = msk.resize((self.size, self.size), resample=Image.NEAREST)
        m_np = (np.array(msk, dtype=np.uint8) > 127).astype(np.float32)
        mask = torch.from_numpy(m_np).unsqueeze(0)
        return img, mask


# ------------------------------
# メトリクス（Dice）
# ------------------------------
@torch.no_grad()
def dice_coeff(pred_logits: torch.Tensor, target_mask: torch.Tensor, eps: float = 1e-6) -> float:
    pred = (torch.sigmoid(pred_logits) > 0.5).float()
    target = (target_mask > 0.5).float()
    inter = (pred * target).sum(dim=[1, 2, 3])
    union = pred.sum(dim=[1, 2, 3]) + target.sum(dim=[1, 2, 3])
    dice = ((2 * inter + eps) / (union + eps)).mean().item()
    return float(dice)


# ------------------------------
# 学習・検証ループ（tqdm 進捗付き）
# ------------------------------

def train_one_epoch(model, loader, optimizer, device):
    model.train()
    criterion = nn.BCEWithLogitsLoss()
    running = 0.0
    n_samples = 0
    pbar = tqdm(loader, desc="Train", leave=False)
    for imgs, masks in pbar:
        imgs = imgs.to(device)
        masks = masks.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(imgs)
        loss = criterion(logits, masks)
        loss.backward()
        optimizer.step()
        bs = imgs.size(0)
        running += loss.item() * bs
        n_samples += bs
        pbar.set_postfix({"loss": f"{running / max(1,n_samples):.4f}"})
    return running / max(1, n_samples)

@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    criterion = nn.BCEWithLogitsLoss()
    loss_sum = 0.0
    n_samples = 0
    dices = []
    pbar = tqdm(loader, desc="Val", leave=False)
    for imgs, masks in pbar:
        imgs = imgs.to(device)
        masks = masks.to(device)
        logits = model(imgs)
        loss = criterion(logits, masks)
        bs = imgs.size(0)
        loss_sum += loss.item() * bs
        n_samples += bs
        dices.append(dice_coeff(logits, masks))
        pbar.set_postfix({"loss": f"{loss_sum / max(1,n_samples):.4f}"})
    avg_loss = loss_sum / max(1, n_samples)
    avg_dice = float(np.mean(dices)) if dices else 0.0
    return avg_loss, avg_dice


# ------------------------------
# 推論結果の保存（白黒 PNG）
# ------------------------------
@torch.no_grad()
def save_predictions(model, loader, device, out_dir: str, max_batches: int = 2):
    os.makedirs(out_dir, exist_ok=True)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    model.eval()
    saved = 0
    for b_idx, (imgs, masks) in enumerate(loader):
        imgs = imgs.to(device)
        masks = masks.to(device)
        logits = model(imgs)
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).float()

        unnorm = torch.clamp(imgs * std + mean, 0.0, 1.0)
        for i in range(imgs.size(0)):
            save_image(unnorm[i], os.path.join(out_dir, f"val{b_idx:02d}_{i:02d}_image.png"))
            pred_np = (preds[i, 0].detach().cpu().numpy() > 0.5).astype(np.uint8) * 255
            Image.fromarray(pred_np, mode="L").save(os.path.join(out_dir, f"val{b_idx:02d}_{i:02d}_pred.png"))
            gt_np = (masks[i, 0].detach().cpu().numpy() > 0.5).astype(np.uint8) * 255
            Image.fromarray(gt_np, mode="L").save(os.path.join(out_dir, f"val{b_idx:02d}_{i:02d}_mask.png"))
        saved += 1
        if saved >= max_batches:
            break


# ------------------------------
# メイン
# ------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--save_root", type=str, default="./runs_unet")
    parser.add_argument("--dataset", type=str, default="duts", choices=["duts","oxford","custom"])
    parser.add_argument("--custom_train_images", type=str, default=None)
    parser.add_argument("--custom_train_masks", type=str, default=None)
    parser.add_argument("--custom_val_images", type=str, default=None)
    parser.add_argument("--custom_val_masks", type=str, default=None)
    parser.add_argument("--custom_val_ratio", type=float, default=0.1)
    parser.add_argument("--encoder_name", type=str, default="resnet34")
    parser.add_argument("--encoder_weights", type=str, default="imagenet")
    parser.add_argument("--seed", type=int, default=42, help="全乱数のシード値")
    args = parser.parse_args()

    # シード固定
    seed_everything(args.seed)

    # 実行時刻でフォルダを作成
    run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.save_root, run_name)
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(os.path.join(run_dir, "val_preds"), exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Info] Using device: {device}")
    print(f"[Info] Seed fixed to {args.seed}")

    # データセット
    if args.dataset == "duts":
        prepare_duts(args.data_dir)
        train_ds = DUTSSaliencyDataset(args.data_dir, split="train", image_size=args.image_size)
        val_ds = DUTSSaliencyDataset(args.data_dir, split="test", image_size=args.image_size)
        print(f"[Info] Using DUTS dataset -> train: {len(train_ds)}, val: {len(val_ds)}")

    elif args.dataset == "custom":
        print("[Info] Using Custom dataset")
        if args.custom_val_images and args.custom_val_masks:
            train_ds = CustomBinarySegDataset(args.custom_train_images, args.custom_train_masks, image_size=args.image_size)
            val_ds = CustomBinarySegDataset(args.custom_val_images, args.custom_val_masks, image_size=args.image_size)
            print(f"[Info] Custom -> train: {len(train_ds)}, val: {len(val_ds)}")
        else:
            full_ds = CustomBinarySegDataset(args.custom_train_images, args.custom_train_masks, image_size=args.image_size)
            val_len = int(len(full_ds) * args.custom_val_ratio)
            train_len = len(full_ds) - val_len
            split_gen = torch.Generator(device="cpu").manual_seed(args.seed)  # ← ここを args.seed に
            train_ds, val_ds = random_split(full_ds, [train_len, val_len], generator=split_gen)
            print(f"[Info] Custom -> train: {train_len}, val: {val_len}")

    else:
        already = oxford_pet_exists(args.data_dir)
        print("[Info] Preparing Oxford-IIIT Pet (download if needed)..." if not already else "[Info] Using existing Oxford-IIIT Pet dataset...")
        full_ds = OxfordPetSegBinary(root=args.data_dir, split="trainval", image_size=args.image_size, download=not already)
        val_len = int(len(full_ds) * args.val_ratio)
        train_len = len(full_ds) - val_len
        split_gen = torch.Generator(device="cpu").manual_seed(args.seed)      # ← ここを args.seed に
        train_ds, val_ds = random_split(full_ds, [train_len, val_len], generator=split_gen)
        print(f"[Info] Oxford-IIIT Pet -> train: {train_len}, val: {val_len}")

    # DataLoader の generator と worker 初期化を固定
    loader_gen = torch.Generator(device="cpu").manual_seed(args.seed)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True,
        worker_init_fn=seed_worker, generator=loader_gen
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
        worker_init_fn=seed_worker, generator=loader_gen
    )

    # モデル
    model = smp.Unet(encoder_name=args.encoder_name, encoder_weights=args.encoder_weights, in_channels=3, classes=1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_dice = -1.0
    best_path = os.path.join(run_dir, "best_unet.pth")
    history = {"train_loss": [], "val_loss": [], "val_dice": []}

    for epoch in range(1, args.epochs + 1):
        print(f"[Info] Epoch {epoch}/{args.epochs}")
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss, val_dice = validate(model, val_loader, device)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_dice"].append(val_dice)

        print(f"[Epoch {epoch:03d}] train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | val_dice={val_dice:.4f}")

        if val_dice > best_dice:
            best_dice = val_dice
            torch.save({
                "model_state": model.state_dict(),
                "dice": best_dice,
                "epoch": epoch,
                "image_size": args.image_size,
                "dataset": args.dataset,
                "encoder_name": args.encoder_name,
                "encoder_weights": args.encoder_weights,
                "seed": args.seed,
            }, best_path)
            print(f"[Info] Saved new best to {best_path} (dice={best_dice:.4f})")

        # 学習曲線の保存
        xs = np.arange(1, len(history["train_loss"]) + 1)
        fig = plt.figure()
        plt.title("Training and Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.plot(xs, history["train_loss"], label="train_loss")
        plt.plot(xs, history["val_loss"], label="val_loss")
        plt.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(run_dir, "loss_curve.png"))
        plt.close(fig)
        fig2 = plt.figure()
        plt.title("Validation Dice")
        plt.xlabel("Epoch")
        plt.ylabel("Dice")
        plt.plot(xs, history["val_dice"], label="val_dice")
        plt.legend()
        fig2.tight_layout()
        fig2.savefig(os.path.join(run_dir, "dice_curve.png"))
        plt.close(fig2)

    if os.path.exists(best_path):
        print("[Info] Loading best checkpoint for visualization...")
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
    vis_dir = os.path.join(run_dir, "val_preds")
    print("[Info] Saving a few validation predictions to disk...")
    save_predictions(model, val_loader, device, vis_dir, max_batches=2)
    print(f"[Done] All outputs are saved under: {run_dir}")


if __name__ == "__main__":
    main()
