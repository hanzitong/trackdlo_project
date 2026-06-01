"""
preprocessing_cdll: ctypes による preprocessing の Python ラッパー

trackdlo_cdll と同じ設計方針。
libpreprocessing_c.so の C 関数を Python から呼べるようにラップする。

[.so のパスを上書きしたい場合]
    import os
    os.environ["PREPROCESSING_LIB_PATH"] = "/path/to/libpreprocessing_c.so"
    import preprocessing_cdll   # ← 環境変数をセットしてから import する
"""

import ctypes
import numpy as np
import os
from pathlib import Path


# ================================================================
# ライブラリのロード
#
# 優先順位:
#   1. 環境変数 PREPROCESSING_LIB_PATH が指すパス
#   2. colcon build ディレクトリ
#   3. colcon install ディレクトリ
# ================================================================

def _load_lib() -> ctypes.CDLL:
    env_path = os.environ.get("PREPROCESSING_LIB_PATH")
    if env_path:
        p = Path(env_path)
        if not p.exists():
            raise FileNotFoundError(
                f"PREPROCESSING_LIB_PATH が指すファイルが見つかりません: {p}"
            )
        return ctypes.CDLL(str(p))

    here    = Path(__file__).resolve()
    ws_root = here.parents[3]  # src/preprocessing_cdll/python/ から3つ上がワークスペースルート
    candidates = [
        ws_root / "build"   / "preprocessing_cdll" / "libpreprocessing_c.so",
        ws_root / "install" / "preprocessing_cdll" / "lib" / "libpreprocessing_c.so",
    ]
    for p in candidates:
        if p.exists():
            return ctypes.CDLL(str(p))
    raise FileNotFoundError(
        "libpreprocessing_c.so が見つかりません。\n"
        "  colcon build --packages-up-to preprocessing_cdll を実行してください。\n"
        "探索したパス:\n" + "\n".join(f"  {p}" for p in candidates)
    )


_lib = _load_lib()


# ================================================================
# 関数シグネチャの登録
#
# ctypes のポインタ型:
#   c_uint8   ↔ uint8_t   (BGR 画像、マスク)
#   c_uint16  ↔ uint16_t  (深度画像 mm)
#   c_double  ↔ double    (点群座標、カメラパラメータ)
#   c_int     ↔ int       (画像サイズ、要素数)
# ================================================================
_u8p  = ctypes.POINTER(ctypes.c_uint8)
_u16p = ctypes.POINTER(ctypes.c_uint16)
_dp   = ctypes.POINTER(ctypes.c_double)
_ip   = ctypes.POINTER(ctypes.c_int)

_lib.prep_images_to_pointcloud.restype  = None
_lib.prep_images_to_pointcloud.argtypes = [
    _u8p, _u16p, _u8p,                    # bgr, depth, mask
    ctypes.c_int, ctypes.c_int,            # rows, cols
    ctypes.c_double, ctypes.c_double,      # fx, fy
    ctypes.c_double, ctypes.c_double,      # cx, cy
    ctypes.c_double,                       # leaf_size
    _dp, _ip,                              # out_pts, out_n
]

_lib.prep_compute_visible_nodes.restype  = None
_lib.prep_compute_visible_nodes.argtypes = [
    _dp, ctypes.c_int,                     # Y, M
    _dp, ctypes.c_int,                     # X, N
    _dp,                                   # geodesic_coord
    ctypes.c_double, ctypes.c_double,      # fx, fy
    ctypes.c_double, ctypes.c_double,      # cx, cy
    ctypes.c_int, ctypes.c_int,            # img_rows, img_cols
    ctypes.c_double,                       # visibility_threshold
    ctypes.c_double,                       # d_vis
    ctypes.c_int,                          # dlo_pixel_width
    _ip, _ip,                              # out_vn, out_vn_len
    _ip, _ip,                              # out_vne, out_vne_len
]


# ================================================================
# 公開 API
# ================================================================

def images_to_pointcloud(
    bgr: np.ndarray,
    depth: np.ndarray,
    mask: np.ndarray,
    fx: float, fy: float, cx: float, cy: float,
    leaf_size: float = 0.005,
) -> np.ndarray:
    """
    BGR 画像 + 深度画像 + マスク → 3D 点群 (N×3, float64, 単位 m)

    bgr    : (H, W, 3) uint8  — OpenCV BGR 形式
    depth  : (H, W)    uint16 — RealSense z16 形式 (mm)
    mask   : (H, W)    uint8  — 0=背景, >0=ケーブル
    fx, fy, cx, cy : カメラ内部パラメータ
    leaf_size : VoxelGrid のセルサイズ (m)。0.0 でダウンサンプリング無効
    """
    rows, cols = bgr.shape[:2]

    # C ABI は行優先の連続メモリを期待する
    # ascontiguousarray: 非連続配列 (スライス等) を連続メモリに変換する
    bgr_c   = np.ascontiguousarray(bgr,   dtype=np.uint8)
    depth_c = np.ascontiguousarray(depth, dtype=np.uint16)
    mask_c  = np.ascontiguousarray(mask,  dtype=np.uint8)

    # 出力バッファ: 最悪ケース = 全ピクセルが点群になる場合
    # rows * cols 点 × 3 座標 = rows*cols*3 個の double
    out_buf = np.empty(rows * cols * 3, dtype=np.float64)
    out_n   = ctypes.c_int(0)

    _lib.prep_images_to_pointcloud(
        bgr_c.ctypes.data_as(_u8p),
        depth_c.ctypes.data_as(_u16p),
        mask_c.ctypes.data_as(_u8p),
        rows, cols,
        fx, fy, cx, cy,
        leaf_size,
        out_buf.ctypes.data_as(_dp),
        ctypes.byref(out_n),
    )

    n = out_n.value
    if n == 0:
        return np.empty((0, 3), dtype=np.float64)
    # out_buf の先頭 n*3 要素を (n, 3) に reshape して返す
    return out_buf[:n * 3].reshape(n, 3)


def compute_visible_nodes(
    Y: np.ndarray,
    X: np.ndarray,
    geodesic_coord: np.ndarray,
    fx: float, fy: float, cx: float, cy: float,
    img_rows: int, img_cols: int,
    visibility_threshold: float = 0.02,
    d_vis: float = 0.05,
    dlo_pixel_width: int = 10,
):
    """
    ノード座標 Y と点群 X から可視ノードリストを計算する。

    Y              : (M, 3) float64 — 前フレームのノード座標
    X              : (N, 3) float64 — 現フレームの点群
    geodesic_coord : (M,)   float64 — 各ノードの累積弧長 [m]
    fx, fy, cx, cy : カメラ内部パラメータ
    img_rows, img_cols : カメラ画像サイズ (画素)
    visibility_threshold : ノード-点群間距離の可視判定閾値 (m)
    d_vis          : extended 側のギャップ許容幅 (m)
    dlo_pixel_width : 投影エッジの描画幅 (画素)

    Returns:
        visible_nodes          : list[int] — 可視ノードのインデックスリスト
        visible_nodes_extended : list[int] — ギャップ補完済みの拡張リスト
    """
    M = len(Y)
    Y_c   = np.ascontiguousarray(Y,              dtype=np.float64)
    X_c   = np.ascontiguousarray(X,              dtype=np.float64)
    geo_c = np.ascontiguousarray(geodesic_coord, dtype=np.float64)

    # 出力バッファ: 最大 M 要素
    out_vn      = np.empty(M, dtype=np.int32)
    out_vn_len  = ctypes.c_int(0)
    out_vne     = np.empty(M, dtype=np.int32)
    out_vne_len = ctypes.c_int(0)

    _lib.prep_compute_visible_nodes(
        Y_c.ctypes.data_as(_dp),   M,
        X_c.ctypes.data_as(_dp),   len(X),
        geo_c.ctypes.data_as(_dp),
        fx, fy, cx, cy,
        img_rows, img_cols,
        visibility_threshold, d_vis, dlo_pixel_width,
        out_vn.ctypes.data_as(_ip),  ctypes.byref(out_vn_len),
        out_vne.ctypes.data_as(_ip), ctypes.byref(out_vne_len),
    )

    vn  = out_vn[:out_vn_len.value].tolist()
    vne = out_vne[:out_vne_len.value].tolist()
    return vn, vne
