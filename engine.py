# -*- coding: utf-8 -*-  # このファイルの文字エンコーディングをUTF-8に指定
# 学習・検証ループ（tqdm 進捗付き）  # 目的：学習用(train)・検証用(validate)ループを提供

from tqdm import tqdm  # 進捗バー表示のためのライブラリtqdmをインポート
import torch  # PyTorch本体
import torch.nn as nn  # ニューラルネット関連モジュール（損失関数など）にアクセス
import numpy as np  # 数値計算（平均など）に使用


def train_one_epoch(model, loader, optimizer, device):
    # 1エポックの学習処理を行う関数（平均損失を返す）

    model.train()  # モデルを学習モードに切替（Dropout/BatchNormが学習挙動）
    criterion = nn.BCEWithLogitsLoss()  # バイナリ用損失：logits（生出力）にSigmoidを内包したBCEを適用
    running = 0.0  # 損失の合計（ミニバッチ毎に加算）
    n_samples = 0  # サンプル数の合計（平均損失計算用の分母）
    pbar = tqdm(loader, desc="Train", leave=False)  # DataLoaderをtqdmでラップして進捗表示

    for imgs, masks in pbar:  # 各ミニバッチについて反復（入力画像と正解マスク）
        imgs = imgs.to(device)  # 入力画像をGPU/CPUデバイスへ転送
        masks = masks.to(device)  # 正解マスクをデバイスへ転送
        optimizer.zero_grad(set_to_none=True)  # 勾配をゼロクリア（set_to_noneでメモリ効率化）
        logits = model(imgs)  # モデルの順伝播（生の出力logitsを得る）
        loss = criterion(logits, masks)  # BCEWithLogitsLossで損失を計算（logitsとターゲットの形状一致が前提）
        loss.backward()  # 損失に基づき逆伝播で勾配を計算
        optimizer.step()  # オプティマイザでパラメータを更新

        bs = imgs.size(0)  # このミニバッチのサンプル数（バッチサイズ）
        running += loss.item() * bs  # 合計損失に（スカラー損失×バッチサイズ）を加算
        n_samples += bs  # サンプル数を加算
        pbar.set_postfix({"loss": f"{running / max(1,n_samples):.4f}"})  # 進捗バー右端に現在の平均損失を表示

    return running / max(1, n_samples)  # 全サンプルに対する平均損失を返す（ゼロ除算防止にmaxを使用）

@torch.no_grad()  # この関数内では勾配を追跡しない（検証は推論のみのため高速化・省メモリ）
def validate(model, loader, device):
    # 検証処理を行う関数（平均損失を返す）

    model.eval()  # モデルを評価モードに切替（Dropout無効・BatchNormが推論挙動）
    criterion = nn.BCEWithLogitsLoss()  # 学習時と同じ損失関数を使用
    loss_sum = 0.0  # 損失の合計（加重平均用にサンプル数で重み付け）
    n_samples = 0  # 検証に用いた総サンプル数

    pbar = tqdm(loader, desc="Val", leave=False)  # 検証用進捗バーを作成
    for imgs, masks in pbar:  # ミニバッチごとに検証
        imgs = imgs.to(device)  # 入力画像をデバイスへ転送
        masks = masks.to(device)  # 正解マスクをデバイスへ転送
        logits = model(imgs)  # 順伝播で出力logitsを得る
        loss = criterion(logits, masks)  # 損失を計算（BCEWithLogitsLoss）

        bs = imgs.size(0)  # ミニバッチ内サンプル数
        loss_sum += loss.item() * bs  # 合計損失に（損失×バッチサイズ）を加算
        n_samples += bs  # 総サンプル数を加算

        pbar.set_postfix({"loss": f"{loss_sum / max(1,n_samples):.4f}"})  # 進捗バー右端に現在の平均損失を表示

    avg_loss = loss_sum / max(1, n_samples)  # 全検証サンプルに対する平均損失を計算
    return avg_loss  # 平均損失を返す
