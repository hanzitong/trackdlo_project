"""
hello_trackdlo.py

trackdlo_cdll (ctypes ラッパー) の最小構成サンプル。
cpp/hello_trackdlo.cpp と同じ処理を Python から行う。

実行:
    # ワークスペースルートで colcon build してから
    colcon build --packages-up-to trackdlo_cdll
    python3 src/trackdlo_examples/python/hello_trackdlo.py
"""

import sys
import math
import numpy as np
from pathlib import Path

# trackdlo_cdll.py の場所をパスに追加する
# このファイルの位置: src/trackdlo_examples/python/
# trackdlo_cdll.py:   src/trackdlo_cdll/python/
_SRC = Path(__file__).resolve().parents[2]   # src/ ディレクトリ
sys.path.insert(0, str(_SRC / "trackdlo_cdll" / "python"))

from trackdlo_cdll import (
    TrackdloState, default_params,
    tracking_step, cpd_lle, sort_pts,
)


def make_cable_cloud(N: int, bend_y: float, seed: int) -> np.ndarray:
    """合成点群を生成する。実際のシステムでは RealSense + preprocessing が担当する。"""
    rng   = np.random.default_rng(seed)
    noise = rng.normal(0.0, 0.005, (N, 3))
    t     = np.linspace(0.0, 1.0, N)
    pts   = np.column_stack([
        t,
        bend_y * np.sin(t * math.pi),
        np.ones(N),
    ])
    return (pts + noise).astype(np.float64)


def initialize(X: np.ndarray, M: int) -> TrackdloState:
    """
    点群 X から TrackdloState を初期化して返す。

    呼び出し側はメモリ管理に注意すること。
    with 文を使うか、finally で state.close() を呼ぶこと。
    詳細: python/memory_management.py を参照。
    """
    state = TrackdloState(M)

    Y_init         = np.linspace(X.min(axis=0), X.max(axis=0), M)
    Y_fit, sig2, _ = cpd_lle(X, Y_init, 0.0,
                              beta=5.0, lambda_=1.0, lle_weight=1.0, mu=0.05)
    Y_sorted       = sort_pts(Y_fit)

    state.Y      = Y_sorted
    state.sigma2 = sig2

    dists = np.linalg.norm(np.diff(Y_sorted, axis=0), axis=1)
    state.geodesic_coord = np.concatenate([[0.0], np.cumsum(dists)])
    state.guide_nodes    = Y_sorted

    return state


def main():
    M      = 10
    N      = 200
    FRAMES = 5

    print("=== hello_trackdlo (Python / ctypes) ===")
    print(f"nodes={M}  points={N}\n")

    print("[frame 0] initializing...")
    X0 = make_cable_cloud(N, bend_y=0.0, seed=0)

    # with 文でメモリを確実に解放する (推奨パターン)
    with initialize(X0, M) as state:
        print(f"  Y[0]   = {state.Y[0]}")
        print(f"  Y[M-1] = {state.Y[M - 1]}")

        params    = default_params()
        all_nodes = list(range(M))

        for f in range(1, FRAMES + 1):
            bend = 0.05 * f
            X    = make_cable_cloud(N, bend_y=bend, seed=f)

            tracking_step(state, X, all_nodes, all_nodes, params)

            print(
                f"[frame {f}]"
                f"  bend={bend:.2f}"
                f"  sigma2={state.sigma2:.6f}"
                f"  length={state.geodesic_coord[-1]:.4f}"
                f"  Y[mid]={state.Y[M // 2]}"
            )

    print("\ndone.")


if __name__ == "__main__":
    main()
