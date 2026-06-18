# =============================================================================
# infer_live_cpu.py  (scripts_aoyama/)
#
# RealSense D405 のカラーストリームを使って
# weights_aoyama/ の学習済み DeepLabV3+ を CPU でリアルタイム推論する。
#
# 事前準備: train.py で weights_aoyama/best_deeplabv3plus_cable.pth を生成すること。
#
# 使い方 (ワークスペースルートから):
#   uv run python src/bmask_gen/scripts_aoyama/infer_live_cpu.py
# =============================================================================

from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs
import torch
import segmentation_models_pytorch as smp

ROOT: Path = Path(__file__).resolve().parent.parent

device: str = "cpu"
print("device =", device)

# ─── モデルのロード ──────────────────────────────────────────────────────
model: smp.DeepLabV3Plus = smp.DeepLabV3Plus(
    encoder_name="resnet34",
    encoder_weights=None,   # 重みは pth から読み込むためダウンロード不要
    in_channels=3,
    classes=1,
    activation=None,
).to(device)

model.load_state_dict(
    torch.load(
        ROOT / "weights_aoyama/best_deeplabv3plus_cable.pth",
        map_location="cpu",
    )
)
model.eval()
print("model loaded")

# ─── RealSense D405 パイプラインの設定 ──────────────────────────────────
# D405 はグローバルシャッター搭載の短距離深度カメラ。
# ここではカラーストリームのみ使用する。
pipeline: rs.pipeline = rs.pipeline()
cfg: rs.config = rs.config()

# カラーストリームを BGR8 フォーマットで有効化
# rs.format.bgr8 を指定すると OpenCV と同じチャンネル順で取得できる
cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

pipeline.start(cfg)
print("RealSense pipeline started")

# ─── 推論ループ ──────────────────────────────────────────────────────────
try:
    while True:
        # wait_for_frames(): 次のフレームセットが届くまでブロックする
        frames: rs.composite_frame = pipeline.wait_for_frames()
        color_frame: rs.video_frame = frames.get_color_frame()

        # カラーフレームが取得できなかった場合はスキップ
        if not color_frame:
            continue

        # RealSense フレームを numpy 配列に変換 (BGR, shape: (480, 640, 3))
        frame: np.ndarray = np.asanyarray(color_frame.get_data())

        # ── 前処理 ───────────────────────────────────────────────────────
        # ※ train.py の __getitem__ と完全に同じ手順・数値にすること。
        #    ここを変えると学習時と入力分布がずれ、推論結果が壊れる。

        # [1] BGR → RGB 変換
        # RealSense SDK は rs.format.bgr8 で取得するため OpenCV と同じ BGR 順。
        # ResNet34 エンコーダは ImageNet (RGB 順) で事前学習されているため
        # RGB に変換する必要がある。train.py も同じ変換をしている。
        rgb: np.ndarray = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # [2] 正規化: uint8 [0, 255] → float32 [0.0, 1.0]
        # ニューラルネットワークへの入力を小さな値に揃えることで学習が安定する。
        #
        # ※ ImageNet 正規化 (mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
        #    は意図的に省略している。train.py も /255.0 のみで学習しているため、
        #    推論でも同じ正規化にしなければならない。
        #    もし推論だけ ImageNet 正規化を加えると入力スケールがずれて
        #    結果が大きく悪化するので注意。
        x: np.ndarray = rgb.astype(np.float32) / 255.0

        # [3] 軸変換: HWC → CHW
        # numpy / OpenCV の画像は (H, W, C) 順 = (高さ, 幅, チャンネル)。
        # PyTorch のモデルは (C, H, W) 順 = (チャンネル, 高さ, 幅) を期待する。
        # np.transpose(x, (2, 0, 1)) は「元の axis2 を先頭に、axis0, axis1 を後ろへ」
        # という軸の入れ替えを意味する。
        #   変換前: shape (480, 640, 3)  ← axis0=H, axis1=W, axis2=C
        #   変換後: shape (3, 480, 640)  ← axis0=C, axis1=H, axis2=W
        x = np.transpose(x, (2, 0, 1))

        # [4] バッチ次元追加: CHW → NCHW
        # PyTorch のモデルは常にバッチ入力 (N, C, H, W) を期待する。
        # リアルタイム推論では 1 枚ずつ処理するため N=1 として先頭に次元を追加する。
        #   変換前: shape (3, 480, 640)     ← (C, H, W)
        #   変換後: shape (1, 3, 480, 640)  ← (N, C, H, W)
        # torch.tensor() は numpy の dtype を引き継ぐ (ここでは float32)。
        x_t: torch.Tensor = torch.tensor(x).unsqueeze(0)

        # ── 推論 ─────────────────────────────────────────────────────────
        with torch.no_grad():
            # model(x_t) の出力形状: (N, classes, H, W) = (1, 1, 480, 640)
            # train.py で activation=None としているため出力は raw logit。
            # sigmoid() で [0, 1] の確率マップに変換する。
            # [0, 0] (多次元インデックス指定) でバッチ次元 N=1 とチャンネル次元 classes=1 を除去(取り出し)し
            # 形状 (480, 640) の 2D 配列にしてから numpy に変換する。
            prob: np.ndarray = torch.sigmoid(model(x_t))[0, 0].numpy()

        # ── マスク生成・表示 ──────────────────────────────────────────────
        # prob > 0.5: 確率が 50% を超えたピクセルをケーブルと判定して 1 にする。
        # astype(np.uint8) は後続の OpenCV 処理 (imshow/addWeighted) に合わせるため。
        mask: np.ndarray = (prob > 0.5).astype(np.uint8)
        print(f"mask pixels (cable): {mask.sum()}", end="\r")

        # (0, 200, 0) は OpenCV BGR 順の緑色。
        # addWeighted で元画像 60% + 緑オーバーレイ 40% を合成する。
        overlay: np.ndarray = frame.copy()
        overlay[mask == 1] = (0, 200, 0)
        result: np.ndarray = cv2.addWeighted(frame, 0.6, overlay, 0.4, 0)

        cv2.imshow("camera", frame)
        cv2.imshow("mask (green = cable)", result)

        if cv2.waitKey(1) & 0xFF == 27:   # ESC で終了
            break

finally:
    # 例外が発生した場合でも必ずパイプラインを停止する
    pipeline.stop()
    cv2.destroyAllWindows()
