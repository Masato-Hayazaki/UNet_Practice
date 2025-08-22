#!/usr/bin/env python3  # 実行時にPython 3でスクリプトを起動するためのシバン
# -*- coding: utf-8 -*-  # ソースコードの文字エンコーディングをUTF-8に指定

import os  # ファイルパス操作やディレクトリ作成などOS依存機能を使うための標準ライブラリ
import argparse  # コマンドライン引数をパースするためのライブラリ
from datetime import datetime  # 現在時刻から実行名（ログ保存用）を生成するために使用
import numpy as np  # 数値演算（シード固定などで使用する可能性あり）
import torch  # PyTorch本体（テンソル・モデル・最適化など）
from torch.utils.data import DataLoader, random_split  # データローダとデータセット分割用ユーティリティ

from data_io import oxford_pet_exists, prepare_duts  # データ存在確認（Oxford）とDUTSデータ準備用の関数をインポート
from datasets import OxfordPetSegBinary, DUTSSaliencyDataset, CustomBinarySegDataset  # 各データセットクラスをインポート
from engine import train_one_epoch, validate  # 1エポックの学習と検証処理を行う関数をインポート
from predict import save_predictions  # 学習済みモデルで推論して画像を保存するユーティリティ
from plotting import save_curves  # 学習・検証の曲線（Loss）を画像保存するユーティリティ
from model import build_unet  # U-Netモデルを構築する関数
from seed_utils import seed_everything, seed_worker  # 乱数シード固定（全体・DataLoaderワーカー）を行う関数

def main():  # スクリプトのエントリポイントとなるmain関数を定義
    parser = argparse.ArgumentParser()  # 引数パーサを作成
    parser.add_argument("--data_dir", type=str, default="./data")  # データのルートディレクトリパス
    parser.add_argument("--image_size", type=int, default=256)  # 入力画像のリサイズ一辺（正方形想定）
    parser.add_argument("--epochs", type=int, default=10)  # 学習エポック数
    parser.add_argument("--batch_size", type=int, default=64)  # バッチサイズ
    parser.add_argument("--lr", type=float, default=1e-4)  # 学習率（Adam用）
    parser.add_argument("--val_ratio", type=float, default=0.1)  # Oxfordデータをtrain/valに分割する際の検証割合
    parser.add_argument("--num_workers", type=int, default=8)  # DataLoaderのワーカープロセス数
    parser.add_argument("--save_root", type=str, default="./result")  # 実行結果（重み・図・予測）を保存するルート
    parser.add_argument("--dataset", type=str, default="duts", choices=["duts","oxford","custom"])  # 使用データセットの指定
    parser.add_argument("--custom_train_images", type=str, default=None)  # カスタムデータセットの訓練用画像ディレクトリ
    parser.add_argument("--custom_train_masks", type=str, default=None)  # カスタムデータセットの訓練用マスクディレクトリ
    parser.add_argument("--custom_val_images", type=str, default=None)  # カスタムデータセットの検証用画像ディレクトリ（省略可）
    parser.add_argument("--custom_val_masks", type=str, default=None)  # カスタムデータセットの検証用マスクディレクトリ（省略可）
    parser.add_argument("--custom_val_ratio", type=float, default=0.1)  # カスタムでvalが未指定の場合のtrain/val分割比
    parser.add_argument("--encoder_name", type=str, default="resnet34")  # U-Netのエンコーダ骨格名（segmentation_models_pytorch想定）
    parser.add_argument("--encoder_weights", type=str, default="imagenet")  # 事前学習重みの種類
    parser.add_argument("--seed", type=int, default=42, help="全乱数のシード値")  # 実験再現性のためのシード
    args = parser.parse_args()  # コマンドライン引数をパースしてargsに格納

    # ★ シード固定
    seed_everything(args.seed)  # Python/NumPy/PyTorch（CPU/GPU）/DataLoader など全体の乱数シードを固定

    # 出力先
    run_name = datetime.now().strftime("%Y%m%d_%H%M%S")  # 現在時刻から一意な実行名（例: 20250820_153000）を作成
    run_dir = os.path.join(args.save_root, run_name)  # 保存先ディレクトリのフルパスを作成
    os.makedirs(run_dir, exist_ok=True)  # 実行用ディレクトリが無ければ作成（存在してもエラーにしない）
    os.makedirs(os.path.join(run_dir, "val_preds"), exist_ok=True)  # 検証時の予測画像保存用サブフォルダを作成

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # CUDAが使えればGPU、なければCPUを選択
    print(f"[Info] Using device: {device}")  # 使用デバイス情報を表示（英語出力指定に沿う）
    print(f"[Info] Seed fixed to {args.seed}")  # 固定したシード値を表示

    # データセット
    if args.dataset == "duts":  # DUTSデータセットを使う分岐
        prepare_duts(args.data_dir)  # DUTSデータの有無を確認し、必要に応じてダウンロード・準備を実施
        train_ds = DUTSSaliencyDataset(args.data_dir, split="train", image_size=args.image_size)  # DUTSの訓練分割データセットを構築
        val_ds = DUTSSaliencyDataset(args.data_dir, split="test", image_size=args.image_size)  # DUTSのテスト分割を検証用として使用
        print(f"[Info] Using DUTS dataset -> train: {len(train_ds)}, val: {len(val_ds)}")  # サンプル数を表示

    elif args.dataset == "custom":  # カスタムデータセットを使う分岐
        print("[Info] Using Custom dataset")  # カスタムデータセット使用を通知
        if args.custom_val_images and args.custom_val_masks:  # 検証用画像・マスクパスが明示的に指定されている場合
            train_ds = CustomBinarySegDataset(args.custom_train_images, args.custom_train_masks, image_size=args.image_size)  # 訓練用カスタムデータセットを構築
            val_ds = CustomBinarySegDataset(args.custom_val_images, args.custom_val_masks, image_size=args.image_size)  # 検証用カスタムデータセットを構築
            print(f"[Info] Custom -> train: {len(train_ds)}, val: {len(val_ds)}")  # サンプル数を表示
        else:  # 検証用が未指定の場合は単一ディレクトリからtrain/valを分割して生成
            full_ds = CustomBinarySegDataset(args.custom_train_images, args.custom_train_masks, image_size=args.image_size)  # 全データを読み込む
            val_len = int(len(full_ds) * args.custom_val_ratio)  # 検証データ数を全体の割合から算出
            train_len = len(full_ds) - val_len  # 訓練データ数を残りから算出
            split_gen = torch.Generator(device="cpu").manual_seed(args.seed)  # 再現性のため分割用乱数生成器にシードを設定
            train_ds, val_ds = random_split(full_ds, [train_len, val_len], generator=split_gen)  # データセットを訓練/検証に分割
            print(f"[Info] Custom -> train: {train_len}, val: {val_len}")  # 分割結果を表示

    else:  # Oxford-IIIT Petデータセットを使う分岐（デフォルト以外の選択肢）
        already = oxford_pet_exists(args.data_dir)  # データが既に存在するかを確認
        print("[Info] Preparing Oxford-IIIT Pet (download if needed)..." if not already else "[Info] Using existing Oxford-IIIT Pet dataset...")  # ダウンロード有無を表示
        full_ds = OxfordPetSegBinary(root=args.data_dir, split="trainval", image_size=args.image_size, download=not already)  # Oxfordデータを読み込み（無ければダウンロード）
        val_len = int(len(full_ds) * args.val_ratio)  # 検証データ数を割合から算出
        train_len = len(full_ds) - val_len  # 訓練データ数を算出
        split_gen = torch.Generator(device="cpu").manual_seed(args.seed)      # 再現性のため分割用乱数生成器にシードを設定
        train_ds, val_ds = random_split(full_ds, [train_len, val_len], generator=split_gen)  # train/valにランダム分割（固定シード）
        print(f"[Info] Oxford-IIIT Pet -> train: {train_len}, val: {val_len}")  # 分割結果を表示

    # ★ DataLoader の generator / worker_init_fn を固定
    loader_gen = torch.Generator(device="cpu").manual_seed(args.seed)  # DataLoaderのシャッフル乱数も固定するためのGenerator
    train_loader = DataLoader(  # 訓練用データローダを作成
        train_ds, batch_size=args.batch_size, shuffle=True,  # シャッフルを有効化してミニバッチをランダム化
        num_workers=args.num_workers, pin_memory=True,  # 並列読み込み数とpin_memory設定でI/Oとホスト→GPU転送を最適化
        worker_init_fn=seed_worker, generator=loader_gen  # 各ワーカーの乱数シードを固定し、生成器を指定
    )
    val_loader = DataLoader(  # 検証用データローダを作成
        val_ds, batch_size=args.batch_size, shuffle=False,  # 検証では順序固定（シャッフルしない）
        num_workers=args.num_workers, pin_memory=True,  # 読み出し並列数とpin_memoryを設定
        worker_init_fn=seed_worker, generator=loader_gen  # ワーカーの乱数シードを固定（再現性確保）
    )

    # モデルと最適化
    model = build_unet(encoder_name=args.encoder_name, encoder_weights=args.encoder_weights).to(device)  # 指定エンコーダでU-Netを構築しデバイスへ転送
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)  # Adam最適化器を学習率指定で作成

    best_path = os.path.join(run_dir, "best_unet.pth")  # ベストモデルの保存先パス
    history = {"train_loss": [], "val_loss": []}  # 学習曲線描画用に履歴を保持

    # 学習ループ
    best_loss = float("inf")  # 追加: 最良(最小)の検証Lossを保持
    for epoch in range(1, args.epochs + 1):  # 1からepochsまでエポックを反復
        print(f"[Info] Epoch {epoch}/{args.epochs}")  # 現在のエポック進捗を表示
        train_loss = train_one_epoch(model, train_loader, optimizer, device)  # 1エポック分の学習を実行し訓練Lossを取得
        val_loss = validate(model, val_loader, device)  # 検証データでのLossを計算

        history["train_loss"].append(train_loss)  # 訓練Lossを履歴に追加
        history["val_loss"].append(val_loss)  # 検証Lossを履歴に追加

        print(f"[Epoch {epoch:03d}] train_loss={train_loss:.4f} | val_loss={val_loss:.4f}")  # エポック結果を整形表示

        if val_loss < best_loss:  # 検証誤差が最小のとき
            best_loss = val_loss  # 最高スコアを更新
            torch.save({  # モデルのベストチェックポイントを保存
                "model_state": model.state_dict(),  # モデル重み（state_dict）
                "best_val_loss": best_loss,  # 追加: 実際の選別指標(最小の検証Loss)
                "epoch": epoch,  # ベスト更新時のエポック
                "image_size": args.image_size,  # 再現に必要なハイパラ（画像サイズ）
                "dataset": args.dataset,  # 使用データセット名
                "encoder_name": args.encoder_name,  # エンコーダ名
                "encoder_weights": args.encoder_weights,  # 事前学習重みの種類
                "seed": args.seed,  # 実験シード
            }, best_path)  # 保存先パスを指定
            print(f"[Info] Saved new best to {best_path} (val_loss={best_loss:.4f})")  # ベスト保存を通知

        save_curves(history, run_dir)  # 現時点までの学習曲線（Loss）を画像として保存

    # 可視化
    if os.path.exists(best_path):  # ベストチェックポイントが存在する場合
        print("[Info] Loading best checkpoint for visualization...")  # 可視化のためにベスト重みを読み込む旨を表示
        ckpt = torch.load(best_path, map_location=device)  # デバイスに合わせてチェックポイントを読み込み
        model.load_state_dict(ckpt["model_state"])  # モデルにベスト重みをロード
    vis_dir = os.path.join(run_dir, "val_preds")  # 検証予測の保存先ディレクトリ
    print("[Info] Saving a few validation predictions to disk...")  # 予測画像の保存開始を通知
    save_predictions(model, val_loader, device, vis_dir, max_batches=2)  # 検証データの一部で推論し、予測マスクを保存（最大2バッチ分）
    print(f"[Done] All outputs are saved under: {run_dir}")  # 実行結果の保存先を最終表示

if __name__ == "__main__":  # このファイルが直接実行された場合のガード
    main()  # main関数を実行
