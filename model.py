# -*- coding: utf-8 -*-  # ソースコードの文字コードをUTF-8に指定
# U-Net モデルのビルダー  # このファイルの目的：U-Netモデルを構築する関数を提供する

import torch  # PyTorchをインポート（モデル操作・学習に必要）

try:
    import segmentation_models_pytorch as smp  # segmentation_models_pytorch (SMP) をインポート
except Exception as e:
    # インポートに失敗した場合、エラーメッセージを出す
    raise ImportError(
        "segmentation_models_pytorch is required. Install via 'pip install segmentation-models-pytorch timm'"
    ) from e
    # segmentation_models_pytorch が無ければ強制的にエラーを発生させ、インストール方法を提示

def build_unet(encoder_name: str = "resnet18", encoder_weights: str = "imagenet"):
    # SMP の U-Net モデルを構築して返す関数
    # encoder_name: エンコーダ部分のネットワーク名（例: resnet34, efficientnet など）
    # encoder_weights: エンコーダの事前学習重み（"imagenet" などを指定可能）

    model = smp.Unet(
        encoder_name=encoder_name,       # U-Netのエンコーダに使う骨格モデルを指定
        encoder_weights=encoder_weights, # 事前学習済み重みを指定（例: ImageNet学習済み）
        in_channels=3,                   # 入力画像のチャネル数（通常RGB=3）
        classes=1                        # 出力クラス数（ここでは1クラス、つまり2値セグメンテーション用）
    )
    return model  # 構築したU-Netモデルを返す
