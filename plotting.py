# -*- coding: utf-8 -*-  # ソースコードの文字コードをUTF-8に指定
# 学習曲線のPNG保存  # このファイルの目的：学習過程のLossをPNG形式で保存する

import os  # ファイルパス操作やディレクトリ操作を行う標準ライブラリ
import numpy as np  # 数値計算や配列操作に使用（エポック数の配列生成など）
import matplotlib.pyplot as plt  # グラフ描画ライブラリ

def save_curves(history: dict, run_dir: str):
    # 学習履歴（lossを含む辞書）を受け取り、グラフを保存する関数

    # loss 曲線 ------------------------------
    xs = np.arange(1, len(history["train_loss"]) + 1)  
    # 横軸（エポック数）を1からNまでの整数で生成
    fig = plt.figure()  # 新しい図を作成
    plt.title("Training and Validation Loss")  # グラフタイトルを設定
    plt.xlabel("Epoch")  # x軸ラベルを「Epoch」に設定
    plt.ylabel("Loss")  # y軸ラベルを「Loss」に設定
    plt.plot(xs, history["train_loss"], label="train_loss")  # 訓練Lossの推移をプロット
    plt.plot(xs, history["val_loss"], label="val_loss")  # 検証Lossの推移をプロット
    plt.legend()  # 凡例を表示（train_lossとval_lossを区別するため）
    fig.tight_layout()  # レイアウトを自動調整（余白が詰まらないように）
    fig.savefig(os.path.join(run_dir, "loss_curve.png"))  
    # 保存先(run_dir)に loss_curve.png としてグラフを保存
    plt.close(fig)  # メモリ解放のため図を閉じる

