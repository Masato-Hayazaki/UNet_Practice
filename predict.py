# -*- coding: utf-8 -*-  # ソースコードの文字エンコーディングをUTF-8に指定
# 推論結果の保存（白黒 PNG）  # このファイルの目的：推論結果をPNG形式で保存する処理をまとめたもの

import os  # ファイル・ディレクトリ操作用
import torch  # PyTorch本体
from torchvision.utils import save_image  # 画像テンソルをPNGなどに保存するためのユーティリティ関数
import numpy as np  # NumPy配列操作（maskを保存する際に使用）
from PIL import Image  # 画像処理ライブラリPillow（マスクをPNGとして保存するのに使用）

@torch.no_grad()  # この関数内では勾配を追跡しない（推論専用処理のため高速化＆メモリ節約）
def save_predictions(model, loader, device, out_dir: str, max_batches: int = 2):
    os.makedirs(out_dir, exist_ok=True)  # 保存先ディレクトリを作成（既に存在してもエラーにしない）

    # 逆正規化用パラメータ（学習時のImageNet正規化を元に戻すためのmeanとstd）
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)  
    # 各チャネルの平均値 (RGB), モデル入力時の正規化パラメータ
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)  
    # 各チャネルの標準偏差, モデル入力時の正規化パラメータ

    model.eval()  # モデルを推論モードに切り替え（DropoutやBatchNormを推論用挙動にする）
    saved = 0  # 保存したバッチ数をカウントする変数

    # DataLoaderから画像とマスクを順に取り出して処理
    for b_idx, (imgs, masks) in enumerate(loader):
        imgs = imgs.to(device)  # 入力画像をGPU/CPUのデバイスへ転送
        masks = masks.to(device)  # 正解マスクもデバイスへ転送

        logits = model(imgs)  # モデルの出力（logits, 生のスコア値）
        probs = torch.sigmoid(logits)  # シグモイド関数で0～1に変換（確率マップ）
        preds = (probs > 0.5).float()  # しきい値0.5で二値化（予測マスク: 0 or 1）

        # 入力画像を「逆正規化」して可視化可能な形式に戻す
        unnorm = torch.clamp(imgs * std + mean, 0.0, 1.0)  
        # (画像×標準偏差 + 平均) により正規化前のRGB値に戻す, 範囲を0~1にクリップ

        # バッチ内の各画像を保存処理
        for i in range(imgs.size(0)):
            # 元画像をPNGで保存
            save_image(unnorm[i], os.path.join(out_dir, f"val{b_idx:02d}_{i:02d}_image.png"))

            # 予測マスクをNumPy配列に変換して保存（0 or 255 の白黒画像に変換）
            pred_np = (preds[i, 0].detach().cpu().numpy() > 0.5).astype(np.uint8) * 255  
            Image.fromarray(pred_np, mode="L").save(os.path.join(out_dir, f"val{b_idx:02d}_{i:02d}_pred.png"))

            # 正解マスクも同様にNumPy配列へ変換して保存（0 or 255）
            gt_np = (masks[i, 0].detach().cpu().numpy() > 0.5).astype(np.uint8) * 255  
            Image.fromarray(gt_np, mode="L").save(os.path.join(out_dir, f"val{b_idx:02d}_{i:02d}_mask.png"))

        saved += 1  # このバッチを保存したのでカウントを増やす
        if saved >= max_batches:  # 保存するバッチ数がmax_batchesに達したら終了
            break
