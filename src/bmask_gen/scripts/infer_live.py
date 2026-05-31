# =============================================================================
# infer_live.py
#
# 学習済み DeepLabV3+ を使ってカメラ映像からリアルタイムでケーブルを検出する。
#
# ─── 推論時の注意点 ──────────────────────────────────────────────────────
#
# 推論は学習とは異なり「重みを更新しない」。そのため:
#   - model.eval()         : BatchNorm / Dropout を推論モードに切り替える
#   - torch.no_grad()      : 計算グラフを構築しない (速度・メモリ改善)
#
# ─── 前処理は学習時と同じにする ─────────────────────────────────────────
#
# モデルは学習時の前処理 (BGR→RGB, /255, HWC→CHW) を前提として重みが最適化されている。
# 推論時に前処理が異なると精度が落ちる。
#
# ─── ビジュアライゼーション ─────────────────────────────────────────────
#
# マスクを緑色オーバーレイとして元画像に重ねて表示する。
# addWeighted で不透明度を調整するため、ケーブル部分が半透明の緑で見える。
#
# =============================================================================

from pathlib import Path

import cv2
import numpy as np
import torch
import segmentation_models_pytorch as smp

# スクリプトの 2 階層上 = bmask_gen/ ルート
ROOT = Path(__file__).resolve().parent.parent

# ─── モデルのロード ──────────────────────────────────────────────────────
device = "cuda" if torch.cuda.is_available() else "cpu"
print("device =", device)

# 推論時はアーキテクチャを再定義してから重みだけを読み込む。
# encoder_weights=None: 事前学習済み重みをダウンロードしない
#   (どうせ直後に pth ファイルで上書きするため不要)
model = smp.DeepLabV3Plus(
    encoder_name="resnet34",
    encoder_weights=None,       # ダウンロード不要 (state_dict で上書きするため)
    in_channels=3,
    classes=1,
    activation=None,
).to(device)

# torch.load(): .pth ファイルからパラメータ辞書を読み込む
# map_location=device: GPU で保存したモデルを CPU でロードする際の変換を自動で行う
# load_state_dict(): モデルに重みを書き込む
model.load_state_dict(
    torch.load(ROOT / "weights/best_deeplabv3plus_cable.pth", map_location=device)
)

# model.eval(): BatchNorm を推論モードに切り替える (必須)
# 学習時は各バッチの平均・分散を使うが、推論時は学習で蓄積した移動平均を使う。
# eval() を呼ばないと BatchNorm の挙動が変わり精度が下がる。
model.eval()
print("model loaded")

# ─── カメラのオープン ────────────────────────────────────────────────────
# cv2.VideoCapture(0): デバイス番号 0 = /dev/video0 を開く
# cv2.CAP_V4L2: Linux の Video4Linux2 バックエンドを明示的に指定する
#   指定しないと自動選択になりフォーマット設定が意図通り効かないことがある
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
if not cap.isOpened():
    raise RuntimeError("カメラを開けませんでした")

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# ─── 推論ループ ──────────────────────────────────────────────────────────
while True:
    # cap.read(): カメラから 1 フレームを取得する
    # 戻り値: (成功フラグ, BGR フレーム画像)
    # C++ の cv::VideoCapture::read() と同じ API
    ret, frame = cap.read()
    if not ret:
        print("フレームを取得できませんでした")
        break

    # ── 前処理 (学習時と完全に同じ手順) ─────────────────────────────────
    # BGR → RGB 変換 (モデルは RGB で学習されているため)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # [0, 255] → [0.0, 1.0] に正規化
    x = rgb.astype(np.float32) / 255.0

    # (H, W, C) → (C, H, W) に軸を並び替え
    x = np.transpose(x, (2, 0, 1))

    # テンソルに変換してバッチ次元を追加: (C,H,W) → (1,C,H,W)
    # .unsqueeze(0): 先頭に次元を挿入する
    # C++ で言えば 3D 配列を 4D 配列の最初の要素として包むイメージ
    x = torch.tensor(x).unsqueeze(0).to(device)

    # ── 推論 ─────────────────────────────────────────────────────────────
    # torch.no_grad(): このブロック内では計算グラフを作らない
    #   → 勾配計算が不要な推論時に使うことでメモリと速度を改善する
    with torch.no_grad():
        # model(x): 順伝播して logit マップを得る (B=1, C=1, H=480, W=640)
        # torch.sigmoid(): logit → 確率 [0, 1] に変換
        # [0, 0]: バッチ次元とチャンネル次元を除去 → (480, 640) の 2D 配列
        # .cpu(): GPU テンソルを CPU に転送
        # .numpy(): PyTorch テンソルを numpy 配列に変換
        prob = torch.sigmoid(model(x))[0, 0].cpu().numpy()

    # ── マスク生成 ────────────────────────────────────────────────────────
    # 確率 > 0.5 のピクセルを「ケーブル」と判定
    # (prob > 0.5) は bool 配列 → astype(np.uint8) で 0/1 に変換
    mask = (prob > 0.5).astype(np.uint8)

    # \r: 行頭に戻って上書きする (ケーブルピクセル数をリアルタイムで表示)
    print(f"mask pixels (cable): {mask.sum()}", end="\r")

    # ── オーバーレイ表示 ──────────────────────────────────────────────────
    # frame をコピーして、ケーブル領域 (mask==1) のピクセルを緑 (BGR: 0,200,0) に塗る
    overlay = frame.copy()
    overlay[mask == 1] = (0, 200, 0)

    # cv2.addWeighted: 2 つの画像を重み付きで合成する
    #   result = frame * 0.6 + overlay * 0.4
    # 結果: ケーブル部分が半透明の緑で表示される
    result = cv2.addWeighted(frame, 0.6, overlay, 0.4, 0)

    cv2.imshow("camera", frame)
    cv2.imshow("mask (green = cable)", result)

    # cv2.waitKey(1): 1ms 待機してキー入力を取得
    # & 0xFF: 上位バイトを除去して ASCII コードを取り出す (Windows 互換のため)
    # 27 = ESC キーのコード
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
