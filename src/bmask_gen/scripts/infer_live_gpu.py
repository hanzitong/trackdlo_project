# =============================================================================
# infer_live_gpu.py
#
# 学習済み DeepLabV3+ を GPU (CUDA) で実行してリアルタイムにケーブルを検出する。
# CPU 版は infer_live_cpu.py を参照。
#
# GPU 版の特徴:
#   - モデルと入力テンソルを GPU メモリに置いて演算する
#   - CPU 版より推論が高速 (特に解像度が大きいほど差が出る)
#   - 推論結果の取り出し時に .cpu() が必要 (GPU→CPU へのコピー)
#
# 動作条件: NVIDIA GPU + CUDA ドライバ + torch の CUDA ビルドが必要
# =============================================================================

from pathlib import Path

import cv2
import numpy as np
import torch
import segmentation_models_pytorch as smp

ROOT: Path = Path(__file__).resolve().parent.parent

# ─── CUDA 確認 ───────────────────────────────────────────────────────────
# このスクリプトは GPU 専用。CUDA が使えない環境では起動時に止める。
if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA が使えません。GPU ドライバと CUDA 対応 PyTorch がインストールされているか確認してください。"
        " CPU で実行する場合は infer_live_cpu.py を使ってください。"
    )

device: str = "cuda"
print("device =", device)

# ─── モデルのロード ──────────────────────────────────────────────────────
# 推論時はアーキテクチャを再定義してから重みだけを読み込む。
# encoder_weights=None: 事前学習済み重みをダウンロードしない
#   (直後に pth ファイルで上書きするため不要)
model: smp.DeepLabV3Plus = smp.DeepLabV3Plus(
    encoder_name="resnet34",
    encoder_weights=None,
    in_channels=3,
    classes=1,
    activation=None,
).to(device)   # モデルのパラメータを GPU メモリに転送する

# torch.load(): .pth ファイルからパラメータ辞書を読み込む
# map_location=device: CPU で保存したモデルを GPU にロードする際の変換を自動で行う
# load_state_dict(): モデルに重みを書き込む
model.load_state_dict(
    torch.load(ROOT / "weights/best_deeplabv3plus_cable.pth", map_location=device)
)

# model.eval(): BatchNorm を推論モードに切り替える (必須)
# 学習時は各バッチの平均・分散を使うが、推論時は学習で蓄積した移動平均を使う。
model.eval()
print("model loaded")

# ─── カメラのオープン ────────────────────────────────────────────────────
cap: cv2.VideoCapture = cv2.VideoCapture(0, cv2.CAP_V4L2)
if not cap.isOpened():
    raise RuntimeError("カメラを開けませんでした")

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# ─── 推論ループ ──────────────────────────────────────────────────────────
while True:
    # cap.read() は (bool, np.ndarray) を返す。デストラクチャリング前に型を宣言する。
    ret: bool
    frame: np.ndarray   # shape (480, 640, 3), dtype uint8, BGR
    ret, frame = cap.read()
    if not ret:
        print("フレームを取得できませんでした")
        break

    # ── 前処理 (学習時と完全に同じ手順) ─────────────────────────────────
    rgb: np.ndarray = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # x は np.ndarray として始まり、torch.tensor() で torch.Tensor に変わる。
    # 型の変わり目を明確にするため、Tensor 変換後は x_t という名前を使う。
    x: np.ndarray = rgb.astype(np.float32) / 255.0
    x = np.transpose(x, (2, 0, 1))                          # (H,W,C) → (C,H,W)
    x_t: torch.Tensor = torch.tensor(x).unsqueeze(0).to(device)  # (C,H,W) → (1,C,H,W) し GPU に転送

    # ── 推論 ─────────────────────────────────────────────────────────────
    with torch.no_grad():
        # .cpu(): 推論結果のテンソルを GPU から CPU メモリにコピーする。
        # numpy() は CPU テンソルにしか使えないため、このコピーが必要。
        prob: np.ndarray = torch.sigmoid(model(x_t))[0, 0].cpu().numpy()

    # ── マスク生成・表示 ──────────────────────────────────────────────────
    mask: np.ndarray = (prob > 0.5).astype(np.uint8)
    print(f"mask pixels (cable): {mask.sum()}", end="\r")

    overlay: np.ndarray = frame.copy()
    overlay[mask == 1] = (0, 200, 0)
    result: np.ndarray = cv2.addWeighted(frame, 0.6, overlay, 0.4, 0)

    cv2.imshow("camera", frame)
    cv2.imshow("mask (green = cable)", result)

    if cv2.waitKey(1) & 0xFF == 27:   # ESC で終了
        break

cap.release()
cv2.destroyAllWindows()
