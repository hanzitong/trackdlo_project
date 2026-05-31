
# =============================================================================
# train_deeplabv3plus_binary.py
#
# DeepLabV3+ を使ってケーブルの「2値セグメンテーション」を学習するスクリプト。
#
# ─── 2値セグメンテーションとは ────────────────────────────────────────────
#
# セグメンテーション = 画像の各ピクセルをクラスに分類するタスク。
# 2値 = クラスが「ケーブル (1)」か「背景 (0)」の2種類のみ。
# → 出力は入力画像と同じ H×W サイズのマスク画像。
#
# ─── DeepLabV3+ アーキテクチャ ────────────────────────────────────────────
#
# エンコーダ (ResNet34) → ASPP → デコーダ → 1チャンネル出力
#
# [エンコーダ]
#   ResNet34 などの CNN で入力画像を圧縮し、特徴マップを抽出する。
#   ImageNet 事前学習済みの重みを使うことで少ないデータでも高精度になる
#   (転移学習 = transfer learning)。
#
# [ASPP (Atrous Spatial Pyramid Pooling)]
#   複数の「穴あき畳み込み (atrous convolution)」で異なるスケールの
#   文脈情報を同時にとらえる。
#   ケーブルのような細長い物体を扱う場合、大きい receptive field が重要。
#
# [デコーダ]
#   圧縮された特徴マップを元の解像度に戻す (アップサンプリング)。
#   エンコーダの浅い層の特徴を skip connection で足すことで
#   エッジ・細部の情報を補う。
#
# [出力 (classes=1, activation=None)]
#   1チャンネルの raw logit マップ (H×W×1)。
#   sigmoid を通すと各ピクセルが「ケーブルである確率」になる。
#   activation=None にしておき、損失関数 BCEWithLogitsLoss に
#   logit のまま渡すほうが数値的に安定するため、モデル側では sigmoid を付けない。
#
# ─── 損失関数 (BCEWithLogitsLoss) ────────────────────────────────────────
#
# BCE = Binary Cross Entropy: 2値分類の標準的な損失関数。
#   正解が 1 のピクセル → -log(sigmoid(logit))
#   正解が 0 のピクセル → -log(1 - sigmoid(logit))
#   これを全ピクセルで平均する。
#
# "WithLogits" の意味:
#   BCELoss(sigmoid(logit), label) と同じ計算だが、
#   数値オーバーフローを防ぐため log と sigmoid を合算して計算している。
#   モデルの出力に sigmoid を付けない理由はここにある。
#
# ─── 評価指標 (IoU) ──────────────────────────────────────────────────────
#
# IoU (Intersection over Union) = 積集合 / 和集合。
#   予測マスクと正解マスクの重なり度合いを表す。
#   0.0 (全く重ならない) ～ 1.0 (完全一致)。
#   セグメンテーションの標準的な精度指標。
#
# =============================================================================

from pathlib import Path    # C++ の std::filesystem::path に相当
import random

import cv2
import numpy as np
from PIL import Image       # PNG 読み込みに使用 (マスク画像の値を正確に読むため)

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import segmentation_models_pytorch as smp   # DeepLabV3+ 実装を提供するライブラリ

# __file__ はこのスクリプト自体のパス。
# .resolve() でシンボリックリンクを解決した絶対パスに変換。
# .parent.parent で scripts/ の 2 つ上、つまりリポジトリルート (bmask_gen/) を指す。
# C++ で言えば: std::filesystem::path(__FILE__).parent_path().parent_path()
ROOT = Path(__file__).resolve().parent.parent


# =============================================================================
# CableDataset クラス
#
# PyTorch の Dataset を継承したデータセットクラス。
# C++ でいう「インターフェースの実装」に相当する。
#
# Dataset は以下の 2 つの純粋仮想関数 (pure virtual function) を要求する:
#   __len__()        : データ数を返す → C++ の size()
#   __getitem__(idx) : idx 番目のサンプルを返す → C++ の operator[]
#
# DataLoader がこのクラスを使って自動的にバッチを作る。
# =============================================================================
class CableDataset(Dataset):
    def __init__(self, image_dir, mask_dir):
        # Path オブジェクトに変換 (文字列のまま渡しても動くが、/ 演算子でパス結合が使える)
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)

        # ディレクトリ内の画像ファイルを列挙してソート。
        # リスト内包表記: [式 for 変数 in イテラブル if 条件]
        # C++ の範囲 for + push_back に相当:
        #   for (auto& p : image_dir) { if (is_image(p)) paths.push_back(p); }
        # .suffix.lower() → 拡張子を小文字で取得 (.PNG → .png)
        self.image_paths = sorted(
            [p for p in self.image_dir.iterdir() if p.suffix.lower() in [".png", ".jpg", ".jpeg"]]
        )

        if len(self.image_paths) == 0:
            raise RuntimeError(f"画像がありません: {self.image_dir}")

    # DataLoader が「何サンプルあるか」を問い合わせるときに呼ばれる
    def __len__(self):
        return len(self.image_paths)

    # DataLoader が idx 番目のサンプルを要求するときに呼ばれる
    # 戻り値: (image_tensor, mask_tensor) のタプル
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        # .stem = 拡張子を除いたファイル名。"img_0001.png" → "img_0001"
        mask_path = self.mask_dir / f"{img_path.stem}.png"

        if not mask_path.exists():
            raise FileNotFoundError(f"対応するマスクがありません: {mask_path}")

        # OpenCV は BGR で読み込むので RGB に変換する
        # (モデルは ImageNet で RGB 学習済みのため、順序が違うと精度が落ちる)
        image = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"画像を読めませんでした: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # PIL で読むと numpy 配列として値が保存される。
        # uint8 で読んで後で float に変換する。
        mask = np.array(Image.open(mask_path), dtype=np.uint8)

        # マスクの値が 0/255 の場合も 0/1 に正規化する
        # (mask > 0) は bool 配列 → astype(float32) で 0.0 / 1.0 に変換
        mask = (mask > 0).astype(np.float32)

        # 学習データのサイズを統一 (サイズが違うとバッチをまとめられない)
        if image.shape[0] != 480 or image.shape[1] != 640:
            raise ValueError(f"画像サイズが 640x480 ではありません: {img_path} shape={image.shape}")
        if mask.shape[0] != 480 or mask.shape[1] != 640:
            raise ValueError(f"マスクサイズが 640x480 ではありません: {mask_path} shape={mask.shape}")

        # ────────────────────────────────────────────────────
        # テンソル形式への変換
        #
        # OpenCV の画像: numpy 配列、形状 (H, W, C) = (480, 640, 3)
        # PyTorch の画像: テンソル、形状 (C, H, W) = (3, 480, 640)
        # この軸の並び替えを「HWC to CHW 変換」と呼ぶ。
        #
        # np.transpose(image, (2, 0, 1)) の意味:
        #   元の軸順 (0=H, 1=W, 2=C) を (2, 0, 1) = (C, H, W) に並び替える。
        #   C++ で言えば 3 次元配列の次元を入れ替えるループに相当する。
        # ────────────────────────────────────────────────────
        image = image.astype(np.float32) / 255.0       # [0, 255] → [0.0, 1.0]
        image = np.transpose(image, (2, 0, 1))          # (H,W,C) → (C,H,W)

        # マスクは (H, W) の 2D → モデルは (1, H, W) を期待するので次元を追加
        # np.expand_dims(x, axis=0): 先頭に次元を追加する
        # C++ で言えば: vector<vector<float>> を vector<vector<vector<float>>> に変換するイメージ
        mask = np.expand_dims(mask, axis=0)             # (H,W) → (1,H,W)

        # numpy 配列を PyTorch テンソルに変換して返す
        return torch.tensor(image, dtype=torch.float32), torch.tensor(mask, dtype=torch.float32)


# =============================================================================
# calc_iou_from_logits
#
# モデル出力 (logit) と正解マスクから IoU を計算する。
# logit は sigmoid を通す前の生の出力値。
#
# 引数:
#   logits    : モデルの生出力 (B, 1, H, W)
#   masks     : 正解マスク (B, 1, H, W)、値は 0.0 か 1.0
#   threshold : sigmoid 出力がこれより大きければ「ケーブル」と判定
#   eps       : ゼロ除算防止の微小値 (分母が 0 になるケースを防ぐ)
# =============================================================================
def calc_iou_from_logits(logits, masks, threshold=0.5, eps=1e-7):
    # sigmoid: logit → 確率 [0, 1] に変換する関数
    probs = torch.sigmoid(logits)

    # 確率 > 0.5 を「ケーブルと予測」と判定し、0 or 1 の float テンソルに変換
    # (probs > threshold) は bool テンソル → .float() で 0.0/1.0 に変換
    preds = (probs > threshold).float()

    # IoU の計算 (ピクセル単位の積集合 / 和集合)
    #   intersection: 予測=1 かつ 正解=1 のピクセル数
    #   union:        予測=1 または 正解=1 のピクセル数
    # dim=(1,2,3) は Batch 以外の全次元 (C,H,W) を集計することを意味する。
    # C++ で言えば image/mask の全要素をループして合計するのと同じ。
    intersection = (preds * masks).sum(dim=(1, 2, 3))
    union = preds.sum(dim=(1, 2, 3)) + masks.sum(dim=(1, 2, 3)) - intersection
    iou = (intersection + eps) / (union + eps)

    # バッチ内の IoU の平均を Python スカラーとして返す
    # .mean() = 平均, .item() = テンソルから Python の float に変換
    return iou.mean().item()


# =============================================================================
# train_one_epoch
#
# 1 エポック (全訓練データを 1 周) の学習を行う。
#
# PyTorch の学習ステップは固定の順番で行う:
#   1. model(images)      → 順伝播 (forward pass)
#   2. criterion(logits)  → 損失計算
#   3. zero_grad()        → 勾配をゼロクリア (前回の勾配が残っているため必須)
#   4. loss.backward()    → 逆伝播 (backward pass) で勾配を計算
#   5. optimizer.step()   → 重みを更新
#
# C++ では自動微分の仕組みがないため手動で勾配を計算するが、
# PyTorch は計算グラフを自動的に構築し backward() で勾配を自動計算する。
# =============================================================================
def train_one_epoch(model, loader, criterion, optimizer, device):
    # model.train(): BatchNorm や Dropout を「学習モード」にする。
    # BatchNorm は学習時はバッチ内の統計量を使い、推論時は蓄積した移動平均を使う。
    # この切り替えを明示的に行う必要がある。
    model.train()
    total_loss = 0.0
    total_iou = 0.0

    # loader はバッチ単位でデータを返すイテレータ。
    # for images, masks in loader: で (image_batch, mask_batch) が得られる。
    # images の形状: (batch_size, 3, 480, 640)
    # masks の形状:  (batch_size, 1, 480, 640)
    for images, masks in loader:
        # .to(device): テンソルを CPU または GPU に転送する。
        # GPU がある場合は device="cuda"、ない場合は device="cpu"。
        images = images.to(device)
        masks = masks.to(device)

        # 順伝播: model(images) → logits (B, 1, H, W)
        logits = model(images)
        loss = criterion(logits, masks)

        # 勾配のゼロクリア (PyTorch は勾配を累積するため、毎ステップ必須)
        optimizer.zero_grad()
        # 逆伝播: loss から各重みへの勾配を自動計算
        loss.backward()
        # 重みの更新 (Adam オプティマイザが勾配を使って重みを調整)
        optimizer.step()

        # .item(): テンソルの値を Python の float に変換する
        # .detach(): このテンソルを計算グラフから切り離す (勾配計算が不要なため)
        total_loss += loss.item()
        total_iou += calc_iou_from_logits(logits.detach(), masks)

    # エポック全体の平均損失と平均 IoU を返す
    return total_loss / len(loader), total_iou / len(loader)


# =============================================================================
# validate_one_epoch
#
# 1 エポックの検証 (validation) を行う。
# 学習とは違い、重みの更新は行わない。
#
# @torch.no_grad() デコレータ:
#   Python の「デコレータ」は C++ の属性 [[nodiscard]] のようなもので、
#   関数に処理を追加する仕組み。
#   torch.no_grad() は「この関数の実行中は計算グラフを構築しない」という指示。
#   勾配が不要な推論・検証時に使うことでメモリ使用量と速度が改善する。
# =============================================================================
@torch.no_grad()
def validate_one_epoch(model, loader, criterion, device):
    # model.eval(): BatchNorm や Dropout を「推論モード」にする
    model.eval()
    total_loss = 0.0
    total_iou = 0.0

    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)

        logits = model(images)
        loss = criterion(logits, masks)

        total_loss += loss.item()
        total_iou += calc_iou_from_logits(logits, masks)

    return total_loss / len(loader), total_iou / len(loader)


# =============================================================================
# main
#
# 学習全体の設定と実行を行うエントリポイント。
# =============================================================================
def main():
    # GPU が使えれば "cuda"、なければ "cpu" を使う
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device =", device)

    # ─── データセットの作成 ───────────────────────────────────────────────
    # CableDataset は images/ と masks/ のペアを管理するだけ (I/O はしない)。
    # 実際の読み込みは DataLoader が __getitem__ を呼ぶときに行われる。
    train_dataset = CableDataset(
        image_dir=ROOT / "data/dataset/images/train",
        mask_dir=ROOT / "data/dataset/masks/train",
    )
    val_dataset = CableDataset(
        image_dir=ROOT / "data/dataset/images/val",
        mask_dir=ROOT / "data/dataset/masks/val",
    )

    # ─── DataLoader の作成 ────────────────────────────────────────────────
    # DataLoader: Dataset から自動的にバッチを作り、並列読み込みするクラス。
    #
    # batch_size=4: 4枚の画像をまとめて1回の学習ステップに使う。
    #   大きいほど学習が安定するが GPU メモリを多く使う。
    #
    # shuffle=True: エポックごとにデータ順をランダムにする。
    #   順序に依存した過学習を防ぐため訓練データでは True が基本。
    #   検証データは順序が関係ないので False でよい。
    #
    # num_workers=2: データ読み込みを 2 つの子プロセスで並列化する。
    #   GPU が計算している間に CPU でデータをロードするため、GPU の待ち時間が減る。
    #   C++ で言えば I/O 専用スレッドを 2 本立てるイメージ。
    #
    # pin_memory=True: ホスト (CPU) メモリをページロック (ピン) する。
    #   ピンされたメモリは DMA で GPU に転送できるため .to(device) が高速になる。
    train_loader = DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=4,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    # ─── モデルの構築 ─────────────────────────────────────────────────────
    # smp = segmentation_models_pytorch ライブラリ
    # DeepLabV3Plus: エンコーダ・デコーダ型のセグメンテーションモデル
    #
    # encoder_name="resnet34":
    #   エンコーダとして ResNet34 を使う。
    #   数字が大きいほど高精度だが重い (resnet50, resnet101 など)。
    #
    # encoder_weights="imagenet":
    #   ImageNet で事前学習済みの重みを使う (転移学習)。
    #   ゼロから学習するより少ないデータで高精度になる。
    #   学習済み重みでエンコーダを初期化し、デコーダはランダム初期化。
    #
    # in_channels=3: 入力は RGB の 3 チャンネル画像
    # classes=1: 出力は「ケーブルである確率」の 1 チャンネルマップ
    # activation=None: sigmoid を付けない (損失関数側で処理するため)
    model = smp.DeepLabV3Plus(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
        activation=None,
    ).to(device)

    # ─── 損失関数とオプティマイザ ──────────────────────────────────────────
    # BCEWithLogitsLoss: 2値分類の標準的な損失関数 (logit を直接受け取る)
    criterion = nn.BCEWithLogitsLoss()

    # Adam: 勾配降下法のアルゴリズム。lr=学習率 (重みを更新する幅)
    # lr=1e-4 は比較的小さい値で、安定した学習が期待できる。
    # model.parameters() = モデルの全重みパラメータを返す
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # ─── 学習ループ ────────────────────────────────────────────────────────
    best_val_iou = -1.0
    num_epochs = 100

    for epoch in range(num_epochs):
        # 訓練データを 1 周して重みを更新
        train_loss, train_iou = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        # 検証データで精度を確認 (重み更新なし)
        val_loss, val_iou = validate_one_epoch(
            model, val_loader, criterion, device
        )

        # f-string: C++ の printf / std::format に相当する文字列フォーマット
        # {変数:.4f} → 小数点以下 4 桁の浮動小数点数で表示
        print(
            f"Epoch {epoch+1:03d}/{num_epochs:03d} "
            f"| train_loss={train_loss:.4f} train_iou={train_iou:.4f} "
            f"| val_loss={val_loss:.4f} val_iou={val_iou:.4f}"
        )

        # 検証 IoU が過去最高を更新したらモデルを保存する (best model 保存)
        # val_loss ではなく val_iou で判断するのは、IoU がより直感的な指標のため。
        if val_iou > best_val_iou:
            best_val_iou = val_iou
            # state_dict(): モデルの重みパラメータを辞書形式で取得する。
            # モデルのアーキテクチャ定義 (クラス) と重みは分けて保存するのが PyTorch の慣習。
            # → ロード時にも同じアーキテクチャを定義してから load_state_dict() で重みを復元する。
            torch.save(model.state_dict(), ROOT / "weights/best_deeplabv3plus_cable.pth")
            print(f"saved best model: val_iou={best_val_iou:.4f}")

    # 最後のエポックの重みも保存 (best と last を使い分ける)
    torch.save(model.state_dict(), ROOT / "weights/last_deeplabv3plus_cable.pth")
    print("saved last model")


# if __name__ == "__main__": ブロック
# Python スクリプトを直接実行したときだけ main() を呼ぶ。
# 他のスクリプトから import されたときは main() を自動実行しない。
# C++ の int main() と同様のエントリポイントだが、import との使い分けのために必要。
if __name__ == "__main__":
    main()
