# =============================================================================
# evaluate.py  (scripts_aoyama/)
#
# 学習済みモデルを val セットで評価し、学習を止めるタイミングの判断材料を提供する。
#
# 出力:
#   1. 数値メトリクス (コンソール)
#      - 画像ごとの IoU / Precision / Recall
#      - val セット全体の mean / min / max / std
#
#   2. 比較画像 (data_aoyama/raw/eval_output/*.png)
#      - 左: 元画像
#      - 中: 正解マスク (白=ケーブル)
#      - 右: 予測マスク overlay (緑=ケーブル)
#      目視でモデルの誤検出・見逃しを確認できる。
#
# 判断の目安:
#   - mean IoU が 0.6 未満    → まだ学習が不十分。epoch を増やすか設定を見直す。
#   - mean IoU が横ばいになった → ほぼ収束。追加学習の効果は薄い。
#   - 数値は良いが目視で変な箇所がある → 過学習や前処理ミスの可能性。
#
# 使い方 (ワークスペースルートから):
#   uv run python src/bmask_gen/scripts_aoyama/evaluate.py
#
# 事前準備:
#   prepare_dataset.py → train.py の順に実行済みであること。
# =============================================================================

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

ROOT: Path = Path(__file__).resolve().parent.parent

WEIGHTS_PATH: Path  = ROOT / "weights_aoyama/best_deeplabv3plus_cable.pth"
VAL_IMG_DIR: Path   = ROOT / "data_aoyama/raw/dataset/images/val"
VAL_MASK_DIR: Path  = ROOT / "data_aoyama/raw/dataset/masks/val"
OUTPUT_DIR: Path    = ROOT / "data_aoyama/raw/eval_output"

# 確率マップをこの閾値でバイナリ化する (train.py と一致させる)
THRESHOLD: float = 0.5


# =============================================================================
# 前処理ユーティリティ
# =============================================================================

def load_image_tensor(img_path: Path) -> tuple[np.ndarray, torch.Tensor]:
    """JPEG 画像を読み込み、元画像 (BGR) とモデル入力テンソルを返す。

    前処理は train.py の __getitem__ と完全に同一にする。

    Returns:
        bgr:   元画像 (480, 640, 3) uint8 BGR  ← 表示用
        x_t:   モデル入力 (1, 3, 480, 640) float32 [0,1]
    """
    # [1] BGR → RGB 変換 (train.py と同じ順序)
    bgr: np.ndarray = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"画像を読めませんでした: {img_path}")
    rgb: np.ndarray = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    # [2] 正規化: uint8 [0,255] → float32 [0,1]
    # train.py と同じく /255.0 のみ。ImageNet mean/std は省略。
    x: np.ndarray = rgb.astype(np.float32) / 255.0

    # [3] 軸変換 HWC→CHW: (480,640,3) → (3,480,640)
    x = np.transpose(x, (2, 0, 1))

    # [4] バッチ次元追加 CHW→NCHW: (3,480,640) → (1,3,480,640)
    x_t: torch.Tensor = torch.tensor(x).unsqueeze(0)

    return bgr, x_t


def load_mask(mask_path: Path) -> np.ndarray:
    """binary PNG マスクを読み込み、0/1 の uint8 配列として返す。

    Returns:
        shape (480, 640), uint8, 値は 0 (背景) または 1 (ケーブル)
    """
    mask: np.ndarray = np.array(Image.open(mask_path), dtype=np.uint8)
    return (mask > 0).astype(np.uint8)


# =============================================================================
# メトリクス計算
# =============================================================================

def compute_metrics(
    pred: np.ndarray,
    gt: np.ndarray,
    eps: float = 1e-7,
) -> dict[str, float]:
    """予測マスクと正解マスクから IoU / Precision / Recall を計算する。

    Args:
        pred: 予測バイナリマスク (H, W), uint8, 値は 0 or 1
        gt:   正解バイナリマスク (H, W), uint8, 値は 0 or 1
        eps:  ゼロ除算防止の微小値

    Returns:
        {"iou": float, "precision": float, "recall": float}

    各指標の意味:
        IoU (Intersection over Union):
            予測と正解の重なり度合い。0=全く外れ、1=完全一致。
            セグメンテーションの主要指標。

        Precision (適合率):
            ケーブルと予測したピクセルのうち本当にケーブルだった割合。
            低い → 背景をケーブルと誤検出している (偽陽性が多い)。

        Recall (再現率):
            実際のケーブルピクセルのうちモデルが検出できた割合。
            低い → ケーブルを見逃している (偽陰性が多い)。
    """
    tp: float = float(np.logical_and(pred == 1, gt == 1).sum())
    fp: float = float(np.logical_and(pred == 1, gt == 0).sum())
    fn: float = float(np.logical_and(pred == 0, gt == 1).sum())

    iou: float       = tp / (tp + fp + fn + eps)
    precision: float = tp / (tp + fp + eps)
    recall: float    = tp / (tp + fn + eps)

    return {"iou": iou, "precision": precision, "recall": recall}


# =============================================================================
# 比較画像の保存
# =============================================================================

def save_comparison(
    bgr: np.ndarray,
    gt_mask: np.ndarray,
    pred_mask: np.ndarray,
    out_path: Path,
    metrics: dict[str, float],
) -> None:
    """元画像・正解マスク・予測 overlay を横並びで保存する。

    Args:
        bgr:       元画像 (480, 640, 3) BGR uint8
        gt_mask:   正解マスク (480, 640) uint8  (0 or 1)
        pred_mask: 予測マスク (480, 640) uint8  (0 or 1)
        out_path:  保存先 PNG パス
        metrics:   IoU / Precision / Recall の値 (画像上に文字で表示)
    """
    h: int
    w: int
    h, w = bgr.shape[:2]

    # ── 正解マスク表示用: グレースケール → BGR に変換して白で表示 ──────────
    gt_vis: np.ndarray = np.zeros((h, w, 3), dtype=np.uint8)
    gt_vis[gt_mask == 1] = (255, 255, 255)

    # ── 予測 overlay: 元画像に緑 (0,200,0) を半透明で合成 ───────────────
    overlay: np.ndarray = bgr.copy()
    overlay[pred_mask == 1] = (0, 200, 0)
    pred_vis: np.ndarray = cv2.addWeighted(bgr, 0.6, overlay, 0.4, 0)

    # ── 3 枚を横に結合 ────────────────────────────────────────────────────
    canvas: np.ndarray = np.hstack([bgr, gt_vis, pred_vis])

    # ── メトリクス文字列を画像左上に描画 ─────────────────────────────────
    iou_str: str = (
        f"IoU={metrics['iou']:.3f}  "
        f"Prec={metrics['precision']:.3f}  "
        f"Rec={metrics['recall']:.3f}"
    )
    cv2.putText(
        canvas, iou_str,
        org=(10, 30),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.7,
        color=(0, 255, 255),   # BGR: 黄緑
        thickness=2,
    )

    # ── カラム見出し ─────────────────────────────────────────────────────
    for col_idx, label in enumerate(["Input", "Ground Truth", "Prediction"]):
        cv2.putText(
            canvas, label,
            org=(col_idx * w + 5, h - 10),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.6,
            color=(200, 200, 200),
            thickness=1,
        )

    cv2.imwrite(str(out_path), canvas)


# =============================================================================
# メイン評価ループ
# =============================================================================

def main() -> None:
    """val セット全体を評価してメトリクスと比較画像を出力するエントリポイント。"""
    # ─── 前提チェック ─────────────────────────────────────────────────────
    if not WEIGHTS_PATH.exists():
        raise FileNotFoundError(
            f"重みファイルが見つかりません: {WEIGHTS_PATH}\n"
            "train.py を先に実行してください。"
        )
    if not VAL_IMG_DIR.exists():
        raise FileNotFoundError(
            f"val 画像ディレクトリが見つかりません: {VAL_IMG_DIR}\n"
            "prepare_dataset.py を先に実行してください。"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ─── モデルロード ─────────────────────────────────────────────────────
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device = {device}")

    model: smp.DeepLabV3Plus = smp.DeepLabV3Plus(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=3,
        classes=1,
        activation=None,
    ).to(device)

    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
    model.eval()
    print(f"重みをロードしました: {WEIGHTS_PATH}")

    # ─── val 画像リストの収集 ─────────────────────────────────────────────
    img_paths: list[Path] = sorted(
        [p for p in VAL_IMG_DIR.iterdir() if p.suffix in [".jpg", ".jpeg"]]
    )
    if len(img_paths) == 0:
        raise RuntimeError(f"val 画像が見つかりません: {VAL_IMG_DIR}")
    print(f"val サンプル数: {len(img_paths)}")

    # ─── 評価ループ ────────────────────────────────────────────────────────
    print()
    print(f"{'画像名':<20}  {'IoU':>6}  {'Precision':>10}  {'Recall':>7}")
    print("-" * 52)

    all_metrics: list[dict[str, float]] = []

    for img_path in img_paths:
        mask_path: Path = VAL_MASK_DIR / f"{img_path.stem}.png"
        if not mask_path.exists():
            print(f"  [WARN] マスクなし: {mask_path.name}")
            continue

        # 前処理
        bgr: np.ndarray
        x_t: torch.Tensor
        bgr, x_t = load_image_tensor(img_path)
        x_t = x_t.to(device)

        gt_mask: np.ndarray = load_mask(mask_path)

        # 推論
        with torch.no_grad():
            # 出力形状: (1, 1, 480, 640) raw logit
            # [0, 0] でバッチ次元と classes 次元を除去 → (480, 640)
            prob: np.ndarray = torch.sigmoid(model(x_t))[0, 0].cpu().numpy()

        pred_mask: np.ndarray = (prob > THRESHOLD).astype(np.uint8)

        # メトリクス計算
        metrics: dict[str, float] = compute_metrics(pred_mask, gt_mask)
        all_metrics.append(metrics)

        # 結果表示
        print(
            f"{img_path.name:<20}  "
            f"{metrics['iou']:>6.3f}  "
            f"{metrics['precision']:>10.3f}  "
            f"{metrics['recall']:>7.3f}"
        )

        # 比較画像を保存
        out_path: Path = OUTPUT_DIR / f"{img_path.stem}_eval.png"
        save_comparison(bgr, gt_mask, pred_mask, out_path, metrics)

    # ─── 統計サマリー ──────────────────────────────────────────────────────
    if len(all_metrics) == 0:
        print("評価できたサンプルがありませんでした。")
        return

    print("-" * 52)

    for key in ["iou", "precision", "recall"]:
        vals: list[float] = [m[key] for m in all_metrics]
        arr: np.ndarray = np.array(vals)
        print(
            f"{key:<10}  "
            f"mean={arr.mean():.3f}  "
            f"min={arr.min():.3f}  "
            f"max={arr.max():.3f}  "
            f"std={arr.std():.3f}"
        )

    print()
    print(f"比較画像を保存しました: {OUTPUT_DIR}/")
    print()

    # ─── 学習継続の判断目安 ─────────────────────────────────────────────────
    mean_iou: float = np.mean([m["iou"] for m in all_metrics])
    if mean_iou < 0.4:
        print(">> mean IoU < 0.4: 学習がまだ十分でない可能性があります。")
        print("   epoch を増やすか、データ数・前処理を見直してください。")
    elif mean_iou < 0.6:
        print(">> mean IoU 0.4〜0.6: 改善の余地があります。")
        print("   eval_output/ の比較画像で誤検出・見逃し箇所を確認してください。")
    else:
        print(">> mean IoU >= 0.6: 実用レベルに近い精度です。")
        print("   eval_output/ の比較画像で残る誤りを確認してください。")


if __name__ == "__main__":
    main()
