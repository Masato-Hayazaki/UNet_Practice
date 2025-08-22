# -*- coding: utf-8 -*-  # このファイルの文字コードをUTF-8に指定
# データセット存在確認＆ダウンロード（DUTS / Oxford）  # 目的：データが存在するか確認し、無ければダウンロードする補助関数群

import os       # ディレクトリ操作（存在確認・作成など）に使用
import zipfile  # ZIPファイルを解凍するために使用
import io       # バイト列をファイル的に扱うために使用（ダウンロードしたZIPを展開する際に利用）
import requests # HTTPリクエストでファイルをダウンロードするために使用

def oxford_pet_exists(root: str) -> bool:
    # Oxford-IIIT Pet データセットが既に存在するか確認する関数
    base = os.path.join(root, "oxford-iiit-pet")  # データセットのベースディレクトリ
    images = os.path.join(base, "images")         # 画像フォルダのパス
    ann = os.path.join(base, "annotations")       # アノテーションフォルダのパス
    return os.path.isdir(images) and os.path.isdir(ann)  
    # 画像とアノテーションのディレクトリが両方存在すればTrueを返す

def duts_exists(root: str) -> bool:
    # DUTS データセットのディレクトリがすでに存在するか確認する関数
    tr_img = os.path.join(root, "DUTS-TR", "DUTS-TR-Image")  # 学習用画像フォルダ
    tr_msk = os.path.join(root, "DUTS-TR", "DUTS-TR-Mask")   # 学習用マスクフォルダ
    te_img = os.path.join(root, "DUTS-TE", "DUTS-TE-Image")  # テスト用画像フォルダ
    te_msk = os.path.join(root, "DUTS-TE", "DUTS-TE-Mask")   # テスト用マスクフォルダ
    return all(os.path.isdir(p) for p in [tr_img, tr_msk, te_img, te_msk])  
    # 4つすべてのディレクトリが存在するならTrue

def download_and_extract_zip(url: str, dest_root: str):
    # URLからZIPファイルをダウンロードして指定ディレクトリに展開する関数
    os.makedirs(dest_root, exist_ok=True)  # 展開先ディレクトリを作成（存在してもエラーにしない）
    print(f"[Info] Downloading: {url}")  # ダウンロード開始メッセージを表示
    r = requests.get(url, stream=True, timeout=60)  # HTTP GETでZIPを取得（タイムアウト60秒）
    r.raise_for_status()  # ステータスコードが200以外なら例外を発生
    z = zipfile.ZipFile(io.BytesIO(r.content))  # ダウンロードしたバイト列をZIPとして読み込む
    z.extractall(dest_root)  # ZIPの中身をdest_rootに展開
    print("[Info] Extracted")  # 展開完了メッセージを表示

def prepare_duts(root: str):
    # DUTS データセットを準備する関数（存在しなければダウンロードして展開）
    if duts_exists(root):  # 既にDUTSが存在するか確認
        print("[Info] DUTS already exists. Skipping download.")  # 存在すればスキップ
        return
    # ダウンロード先URL（公式サイトからの直リンク）
    train_url = "http://saliencydetection.net/duts/download/DUTS-TR.zip"  # 訓練データ用ZIP
    test_url  = "http://saliencydetection.net/duts/download/DUTS-TE.zip"  # テストデータ用ZIP
    download_and_extract_zip(train_url, root)  # 訓練用ZIPをダウンロードして展開
    download_and_extract_zip(test_url, root)   # テスト用ZIPをダウンロードして展開
    print("[Info] DUTS prepared.")  # 準備完了メッセージを表示
