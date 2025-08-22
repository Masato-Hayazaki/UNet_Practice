# -*- coding: utf-8 -*-  # このファイルの文字コードをUTF-8に指定
# 乱数シード固定用ユーティリティ  # ファイル全体の用途を示すコメント

import os      # OS環境変数の操作のためにインポート
import random  # Python組み込みの乱数生成用モジュール
import numpy as np  # NumPyの乱数生成・数値計算を利用するため
import torch   # PyTorch本体を利用するため

def seed_everything(seed: int):
    # 乱数の再現性を確保するために、各ライブラリの乱数シードを固定する関数

    os.environ["PYTHONHASHSEED"] = str(seed)  # Pythonのハッシュ乱数シードを固定（再現性確保のため）
    random.seed(seed)  # Python標準ライブラリ random のシードを設定
    np.random.seed(seed)  # NumPyの乱数シードを設定
    torch.manual_seed(seed)  # PyTorch（CPU上）の乱数シードを設定
    torch.cuda.manual_seed_all(seed)  # PyTorch（全GPU）の乱数シードを設定

    # cuDNN の決定論的動作を強制
    torch.backends.cudnn.benchmark = False  # 入力サイズに応じた最適アルゴリズム探索を無効化（再現性のため）
    torch.backends.cudnn.deterministic = True  # 決定論的アルゴリズムを強制（結果の再現性を担保）

    # 可能であればPyTorchにおいて「決定論的アルゴリズムのみ」を使用
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)  
        # Trueに設定すると非決定論的アルゴリズムを禁止
        # warn_only=True の場合、非決定論的なものが呼ばれた時に警告を出すだけで例外は出さない
    except Exception:  
        pass  # 古いPyTorchではこの関数が存在しない可能性があるため例外は無視

def seed_worker(worker_id: int):
    # DataLoaderのworkerプロセスごとに乱数を固定するための関数

    worker_seed = (torch.initial_seed() + worker_id) % (2**32)  
    # 各workerに異なるシードを与えるため、PyTorchの初期シード＋worker_idを基に計算し32bitに収める

    np.random.seed(worker_seed)  # NumPyのシードを設定
    random.seed(worker_seed)  # Python標準randomのシードを設定
