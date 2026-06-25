

import numpy as np
import torch
import cv2


from trackdlo_cdll import (
    TrackdloState,
    TdloParams,
    default_params,
    tracking_step,
    cpd_lle,
    sort_pts,
)
# from preprocessing_cdll import images_to_pointcloud, compute_visible_nodes



"""
Trackdlo  helper functions

"""

NUM_NODES: int   = 15
# バグ修正 (cpd_lle 異常大座標値): 深度外れ値が Y_init スケールを狂わせるため上限を設ける
# 詳細: docs/problem_solving/cpd_lle_abnormal_large_value.md
Z_MAX_M: float = 2.0

IMAGENET_MEAN: np.ndarray = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD: np.ndarray  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# =============================================================================
# ヘルパー関数
# =============================================================================

def initialize_state(X: np.ndarray) -> "TrackdloState | None":
    """点群 X から TrackdloState を初期化して返す。失敗時は None。"""
    state: TrackdloState = TrackdloState(NUM_NODES)

    # バグ修正 (cpd_lle 異常大座標値): Z 外れ値を除去しないと Y_init が巨大スケールになり
    # C++ 内の 0.1m フィルタで全点が除外され、cpd_lle が動かないまま収束を返す
    # 詳細: docs/problem_solving/cpd_lle_abnormal_large_value.md
    X = X[X[:, 2] < Z_MAX_M]
    if len(X) < NUM_NODES * 3:
        print(f"not enough valid points after Z filter: {len(X)}")
        state.close()
        return None

    Y_init: np.ndarray = np.linspace(X.min(axis=0), X.max(axis=0), NUM_NODES)

    Y_fit: np.ndarray
    sigma2: float
    converged: bool
    Y_fit, sigma2, converged = cpd_lle(
        X, Y_init, sigma2=0.0,
        beta=5.0, lambda_=1.0, lle_weight=1.0, mu=0.05,
    )
    print(f"  [cpd_lle] converged={converged}  sigma2={sigma2:.6f}")

    if not np.all(np.isfinite(Y_fit)):
        print("  [cpd_lle] NaN/Inf detected, skipping initialization")
        state.close()
        return None
    # バグ修正 (cpd_lle 異常大座標値): isfinite は有限な異常大値を見逃すため境界チェックを追加
    if np.any(np.abs(Y_fit) > Z_MAX_M * 10):
        print(f"  [cpd_lle] abnormally large value detected: max={np.abs(Y_fit).max():.2f}")
        state.close()
        return None

    Y_sorted: np.ndarray = sort_pts(Y_fit)
    if not np.all(np.isfinite(Y_sorted)):
        print("  [sort_pts] NaN/Inf detected, skipping initialization")
        state.close()
        return None
    # バグ修正 (sort_pts 後の異常大座標値): sort_pts は最近傍順にノードを並べ替えるが、
    # cpd_lle が端ノードを点群から遠い位置に残した場合、並べ替え後に中間インデックスへ
    # 移動することがある。get_mid_node はこのインデックスを返すため、
    # sort_pts 後にも異常値チェックが必要。
    if np.any(np.abs(Y_sorted) > Z_MAX_M * 10):
        print(f"  [sort_pts] abnormally large value detected: max={np.abs(Y_sorted).max():.2f}")
        state.close()
        return None

    state.Y      = Y_sorted
    state.sigma2 = sigma2

    dists: np.ndarray = np.linalg.norm(np.diff(Y_sorted, axis=0), axis=1)
    state.geodesic_coord = np.concatenate([[0.0], np.cumsum(dists)])
    state.guide_nodes    = Y_sorted
    return state


def get_mid_node(Y: np.ndarray) -> np.ndarray:
    """ノード列 Y の中間インデックスの座標 (3,) を返す。"""
    # return Y[0]
    return Y[len(Y) // 2]


def _project_pt(x3: float, y3: float, z3: float) -> "tuple[int, int] | None":
    """3D 点をピンホールモデルで 2D 画素座標に変換する。z3<=0 なら None。"""
    if not (np.isfinite(z3) and z3 > 0):
        return None
    return int(x3 * FX / z3 + CX), int(y3 * FY / z3 + CY)


def _draw_coord_axes(img: np.ndarray, origin: np.ndarray) -> None:
    """カメラ座標系の XY 軸を img に in-place で描画する。"""
    ox: float = float(origin[0])
    oy: float = float(origin[1])
    oz: float = float(origin[2])

    origin_px: "tuple[int, int] | None" = _project_pt(ox, oy, oz)
    if origin_px is None:
        return

    L: float = oz * 0.1

    axes: list[tuple[float, float, float, tuple[int, int, int], str]] = [
        (ox + L, oy, oz, (0,   0, 255), "X"),
        (ox, oy + L, oz, (0, 255,   0), "Y"),
    ]

    tx: float
    ty: float
    tz: float
    color: tuple[int, int, int]
    label: str
    for tx, ty, tz, color, label in axes:
        tip_px: "tuple[int, int] | None" = _project_pt(tx, ty, tz)
        if tip_px is None:
            continue
        cv2.arrowedLine(img, origin_px, tip_px, color, 2, tipLength=0.25)
        # lx: int = int(tip_px[0] + (tip_px[0] - origin_px[0]) * 0.2)
        # ly: int = int(tip_px[1] + (tip_px[1] - origin_px[1]) * 0.2)
        # cv2.putText(img, label, (lx, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


def draw_debug(
    bgr: np.ndarray,
    depth_mm: np.ndarray,
    mask: np.ndarray,
    Y: "np.ndarray | None",
) -> None:
    """デバッグ用 3 ウィンドウを cv2.imshow で更新する。"""
    vis_bgr: np.ndarray = bgr.copy()
    if Y is not None:
        prev_px: "tuple[int, int] | None" = None
        h: int = 0
        w: int = 0
        h, w = vis_bgr.shape[:2]    # for preventing cv2.circle() error
        x3: float
        y3: float
        z3: float
        for x3, y3, z3 in Y:
            if not (np.isfinite(x3) and np.isfinite(y3) and np.isfinite(z3) and z3 > 0):
                continue
            px: int = int(x3 * FX / z3 + CX)
            py: int = int(y3 * FY / z3 + CY)
            if not (0 <= px < w and 0 <= py < h):   # for preventing cv2.circle() error
                prev_py = None
                continue
            cv2.circle(vis_bgr, (px, py), 5, (0, 0, 255), -1)
            if prev_px is not None:
                cv2.line(vis_bgr, prev_px, (px, py), (0, 220, 255), 2)
            prev_px = (px, py)
    cv2.imshow("BGR image", vis_bgr)

    depth_vis: np.ndarray = np.clip(depth_mm, 0, 10000).astype(np.float32)
    depth_vis = (depth_vis / 10000.0 * 255.0).astype(np.uint8)
    # cv2.imshow("depth image", depth_vis)

    overlay: np.ndarray      = bgr.copy()
    overlay[mask == 1]       = (0, 255, 0)
    mask_overlay: np.ndarray = cv2.addWeighted(bgr, 0.6, overlay, 0.4, 0)
    _draw_coord_axes(mask_overlay, np.array([0.0, 0.0, 0.3]))
    cv2.imshow("mask overlay", mask_overlay)





