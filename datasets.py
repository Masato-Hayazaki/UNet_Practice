# -*- coding: utf-8 -*-  # このファイルの文字コード（UTF-8）を指定
# 2値セグメンテーション用の Dataset 群  # 目的：バイナリ（前景/背景）セグメンテーション向けのデータセット定義

import os  # パス結合やファイル列挙などのOS操作に使用
from typing import Tuple  # 関数の戻り値型ヒント（タプル）に使用
import numpy as np  # マスクの二値化など数値配列処理に使用
from PIL import Image  # 画像の読み込み・型変換（RGB/L）に使用

import torch  # テンソル変換や学習用に使用
from torch.utils.data import Dataset  # PyTorchのデータセット基底クラス
import torchvision.transforms as T  # 画像のリサイズ・テンソル化・正規化の変換群
from torchvision.datasets import OxfordIIITPet  # Oxford-IIIT Pet データセットの公式ラッパー

class CustomBinarySegDataset(Dataset):
    """
    任意のフォルダ構成から2値セグメンテーションの (image, mask) を読み込むDataset
    - 画像: RGBに変換して正規化
    - マスク: グレースケールを 0/1 に二値化（>127を前景=1）
    """  # このクラスの目的・入出力仕様を説明するドキュメンテーション文字列
    def __init__(self, img_dir, mask_dir, image_size=256,
                 img_exts=(".jpg", ".jpeg", ".png", ".bmp"),
                 threshold=127):
        super().__init__()  # Dataset 基底クラスの初期化
        self.img_dir = img_dir  # 画像ディレクトリのパスを保持
        self.mask_dir = mask_dir  # マスクディレクトリのパスを保持
        self.size = image_size  # 画像とマスクのリサイズ解像度（正方形）
        self.threshold = threshold  # マスク二値化の閾値（0-255のうち）
        # 画像のstem一覧を作成（拡張子違いを吸収）
        names = []  # ファイル名から拡張子を除いた「stem」を格納するリスト
        for f in os.listdir(img_dir):  # 画像ディレクトリ内のファイルを列挙
            if f.lower().endswith(img_exts) and not f.startswith('.'):  # 指定拡張子かつ隠しファイルでないものを対象
                names.append(os.path.splitext(f)[0])  # 拡張子を外したファイル名（stem）を追加
        self.names = sorted(names)  # 再現性のためstemをソートして保持

        # 画像変換（学習時と同じ正規化）
        self.img_tf = T.Compose([  # 画像前処理を順に実行するパイプライン
            T.Resize((image_size, image_size), interpolation=T.InterpolationMode.BILINEAR),  # 画像を双一次補間でリサイズ
            T.ToTensor(),  # [H,W,C]→[C,H,W]かつ[0,255]→[0,1]にスケーリング
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # ImageNet統計でチャネル正規化
        ])

    def __len__(self):
        return len(self.names)  # データセットのサンプル数（stem数）を返す

    def _open_image(self, stem):
        # 画像ファイルを探す（拡張子違い対応）
        for ext in [".jpg", ".jpeg", ".png", ".bmp"]:  # 許可する拡張子を順に確認
            p = os.path.join(self.img_dir, stem + ext)  # 画像ファイルの候補パスを作成
            if os.path.isfile(p):  # 実在すれば
                return Image.open(p).convert("RGB")  # 画像を開いてRGBへ変換して返す
        raise FileNotFoundError(f"Image not found for {stem}")  # 見つからない場合は例外

    def _open_mask(self, stem):
        # マスクファイル（PNGなどグレースケール推奨）
        for ext in [".png", ".jpg", ".bmp"]:  # マスクの拡張子候補
            p = os.path.join(self.mask_dir, stem + ext)  # マスクファイルの候補パスを作成
            if os.path.isfile(p):  # 実在すれば
                return Image.open(p).convert("L")  # マスクを開いてグレースケール（L）へ変換して返す
        raise FileNotFoundError(f"Mask not found for {stem}")  # 見つからない場合は例外

    def __getitem__(self, idx):
        stem = self.names[idx]  # インデックスに対応するstem（ベース名）を取得
        img = self._open_image(stem)  # stemに対応する画像を開く（RGB）
        mask = self._open_mask(stem)  # stemに対応するマスクを開く（L）

        # リサイズ（マスクはNEARESTで段差保持）
        img = self.img_tf(img)  # 画像に前処理（Resize→Tensor→Normalize）を適用
        mask = mask.resize((self.size, self.size), resample=Image.NEAREST)  # マスクを最近傍補間でリサイズ（境界の段差保持）

        # 0/255 → 0/1 に二値化
        m_np = (np.array(mask, dtype=np.uint8) > self.threshold).astype(np.float32)  # 閾値で二値化してfloat32へ
        m = torch.from_numpy(m_np).unsqueeze(0)  # [H,W]→[1,H,W]に次元追加しTensor化（チャネル次元を確保）

        return img, m  # 前処理済み画像テンソルと二値マスクテンソルを返す


class OxfordPetSegBinary(Dataset):
    # Oxford-IIIT Pet のラベル {1:前景,2:境界,3:背景} を 2値化  # クラスの目的を説明
    def __init__(self, root: str, split: str = "trainval", image_size: int = 256, download: bool = True):
        super().__init__()  # 基底クラス初期化
        self.base = OxfordIIITPet(root=root, split=split, download=download, target_types=("segmentation",))
        # torchvisionのOxfordIIITPetを内部に保持（画像と対応セグメンテーションを取得）
        self.img_tf = T.Compose([  # 入力画像の前処理
            T.Resize((image_size, image_size), interpolation=T.InterpolationMode.BILINEAR),  # 画像を双一次補間でリサイズ
            T.ToTensor(),  # テンソル化＆[0,1]スケーリング
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # ImageNet正規化
        ])
        self.mask_tf = T.Compose([  # マスクの前処理
            T.Resize((image_size, image_size), interpolation=T.InterpolationMode.NEAREST),  # 最近傍補間でリサイズ（ラベル保持）
        ])

    def __len__(self) -> int:
        return len(self.base)  # 内部のOxfordIIITPetデータ数を返す

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img, target = self.base[idx]  # 画像（PIL RGB）とセグメンテーション（PIL L/パレット）を取得
        img = self.img_tf(img)  # 画像に前処理（Resize→Tensor→Normalize）を適用
        mask_pil: Image.Image = self.mask_tf(target)  # マスクを最近傍補間でリサイズ（PIL画像のまま）
        mask_np = np.array(mask_pil, dtype=np.uint8)  # NumPy配列に変換（ラベル値 1/2/3）
        # {1: 前景, 2: 境界, 3: 背景} → 2値（背景のみ0、それ以外1）
        bin_np = np.where(mask_np == 3, 0, 1).astype(np.float32)  # 背景(3)→0, それ以外→1 に変換
        mask = torch.from_numpy(bin_np).unsqueeze(0)  # [H,W]→[1,H,W]に変換しTensor化
        return img, mask  # 前処理済み画像テンソルと二値マスクを返す


class DUTSSaliencyDataset(Dataset):
    # DUTS は既に白黒バイナリマスク（0/255）。0/1 に変換して返す。  # クラスの目的を説明
    def __init__(self, root: str, split: str = "train", image_size: int = 256):
        super().__init__()  # 基底クラス初期化
        self.split = split  # "train" or "test"（"TE"側は検証用途）
        base = "DUTS-TR" if split == "train" else "DUTS-TE"  # サブディレクトリ名を分岐
        self.img_dir = os.path.join(root, base, f"{base}-Image")  # 画像ディレクトリのフルパス
        self.msk_dir = os.path.join(root, base, f"{base}-Mask")  # マスクディレクトリのフルパス
        # 画像 stem 一覧
        self.names = sorted([os.path.splitext(f)[0] for f in os.listdir(self.img_dir) if not f.startswith('.')])
        # 画像ファイル名から拡張子を除いたstemを列挙し、隠しファイルを除外してソート保存
        self.img_tf = T.Compose([  # 画像の前処理パイプライン
            T.Resize((image_size, image_size), interpolation=T.InterpolationMode.BILINEAR),  # 双一次補間でリサイズ
            T.ToTensor(),  # テンソル化＆[0,1]へ
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # ImageNet正規化
        ])
        self.size = image_size  # マスクのリサイズにも使用するサイズ

    def __len__(self):
        return len(self.names)  # データ数（stem数）を返す

    def _open_mask(self, stem: str) -> Image.Image:
        for ext in [".png", ".jpg", ".bmp"]:  # マスクの拡張子候補を順に確認
            p = os.path.join(self.msk_dir, stem + ext)  # 候補パス作成
            if os.path.isfile(p):  # 実在すれば
                return Image.open(p).convert("L")  # マスクを開いてグレースケール化
        raise FileNotFoundError(f"Mask not found for {stem}")  # 見つからない場合は例外

    def __getitem__(self, idx: int):
        stem = self.names[idx]  # 対象サンプルのstemを取得
        img_path = None  # 見つかった画像のフルパスを入れる変数
        for ext in [".jpg", ".png", ".bmp"]:  # 画像拡張子の候補を順に確認
            p = os.path.join(self.img_dir, stem + ext)  # 画像の候補パス
            if os.path.isfile(p):  # 実在すれば
                img_path = p  # 採用
                break  # 最初に見つかったものを使用
        if img_path is None:  # どの拡張子でも見つからなかった場合
            raise FileNotFoundError(f"Image not found for {stem}")  # 例外を送出
        img = Image.open(img_path).convert("RGB")  # 画像を開いてRGBに変換
        msk = self._open_mask(stem)  # 対応するマスクを開く

        img = self.img_tf(img)  # 画像に前処理（Resize→Tensor→Normalize）を適用
        msk = msk.resize((self.size, self.size), resample=Image.NEAREST)  # マスクを最近傍補間でリサイズ（ラベル保持）
        m_np = (np.array(msk, dtype=np.uint8) > 127).astype(np.float32)  # 0/255を閾値127で0/1に二値化
        mask = torch.from_numpy(m_np).unsqueeze(0)  # [H,W]→[1,H,W]にしてTensor化

        return img, mask  # 前処理済み画像テンソルと二値マスクテンソルを返す
