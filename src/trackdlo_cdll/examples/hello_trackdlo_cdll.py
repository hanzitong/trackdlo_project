"""
hello_trackdlo_cdll.py

trackdlo_cdll (ctypes ラッパー) の最小構成サンプル。
pure_trackdlo の examples/hello_trackdlo.cpp と同じ処理を Python から行う。

実行:
    # ワークスペースルートから
    colcon build --packages-up-to trackdlo_cdll
    python3 src/trackdlo_cdll/examples/hello_trackdlo_cdll.py
"""

import sys
import math
import numpy as np
from pathlib import Path

# trackdlo_cdll.py のあるディレクトリを検索パスに追加する
# (インストールせずにリポジトリから直接実行するための措置)
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from trackdlo_cdll import (
    TrackdloState, TdloParams, default_params,
    tracking_step, cpd_lle, sort_pts,
)


# ============================================================
# 点群生成: バイナリマスク相当
#
# ケーブルを X 軸に沿った直線としてモデル化し、
# ガウシアンノイズを加えて実際のマスクから得られる点群を模倣する。
# bend_y を 0 以外にすると Y 方向に曲げた形状を作れる。
# ============================================================
def make_cable_cloud(N: int, bend_y: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, 0.005, (N, 3))   # 5mm 標準偏差
    t = np.linspace(0.0, 1.0, N)

    pts = np.zeros((N, 3))
    pts[:, 0] = t                                   # X: 0 → 1
    pts[:, 1] = bend_y * np.sin(t * math.pi)        # Y: 半円弧状に曲げる
    pts[:, 2] = 1.0                                  # Z: カメラから 1m 先
    return (pts + noise).astype(np.float64)


# ============================================================
# 初期化
#
# 点群 X から TrackdloState を作る。
# 1. cpd_lle で初期ノード Y_init を点群にフィットさせる
# 2. sort_pts でノードをケーブル沿いに順序付ける
# 3. geodesic_coord (各ノードの累積弧長) を計算する
#    ← tracking_step の呼び出し前に必須
# 4. guide_nodes をセットする (前フレームの可視ノード)
# ============================================================
def initialize(X: np.ndarray, M: int) -> TrackdloState:
    state  = TrackdloState(M)

    # 点群の端点を結ぶ直線上に M 点の初期ノードを配置する
    Y_init = np.linspace(X.min(axis=0), X.max(axis=0), M)

    # cpd_lle: 初期ノードを点群にフィットさせる
    # (reg() は Y_init を無視して原点から始めるため、ここでは使わない)
    Y_fit, sigma2, ok = cpd_lle(
        X, Y_init, 0.0,
        beta=5.0, lambda_=1.0, lle_weight=1.0, mu=0.05,
    )
    print(f"  cpd_lle converged: {ok}")

    # sort_pts: 最近傍グリーディ探索でノードをケーブル沿いに並べ替える
    Y_sorted = sort_pts(Y_fit)

    state.Y      = Y_sorted
    state.sigma2 = sigma2

    # geodesic_coord: 隣接ノード間距離の累積和
    # tracking_step の内部で traverse_euclidean が参照するため必須
    diffs  = np.diff(Y_sorted, axis=0)
    dists  = np.linalg.norm(diffs, axis=1)
    state.geodesic_coord = np.concatenate([[0.0], np.cumsum(dists)])

    # guide_nodes: 前フレームの可視ノード。初期化時は現在のノードをセット。
    state.guide_nodes = Y_sorted

    return state


def main():
    M      = 10   # トラッキングノード数
    N      = 200  # 点群の点数
    FRAMES = 5    # トラッキングフレーム数

    print("=== hello_trackdlo_cdll ===")
    print(f"nodes={M}  points={N}\n")

    # --------------------------------------------------------
    # Step 1: 初期フレームの点群を作り、状態を初期化する
    # --------------------------------------------------------
    print("[frame 0] initializing...")
    X0    = make_cable_cloud(N, bend_y=0.0, seed=0)
    state = initialize(X0, M)
    print(f"  Y[0]   = {state.Y[0]}")
    print(f"  Y[M-1] = {state.Y[M - 1]}")

    # --------------------------------------------------------
    # Step 2: パラメータ設定
    # --------------------------------------------------------
    params = default_params()
    # 必要に応じて調整:
    # params.beta    = 5.0
    # params.lambda_ = 1.0   # lambda は Python 予約語なので lambda_
    # params.mu      = 0.05

    # --------------------------------------------------------
    # Step 3: トラッキングループ
    #
    # visible_nodes: オクルージョンなしの場合は全ノードを渡す。
    # 実際のシステムでは preprocessing::compute_visible_nodes() で取得する。
    # --------------------------------------------------------
    all_nodes = list(range(M))

    for f in range(1, FRAMES + 1):
        bend  = 0.05 * f
        X     = make_cable_cloud(N, bend_y=bend, seed=f)

        # トラッキング: state.Y と state.sigma2 が in-place で更新される
        tracking_step(state, X, all_nodes, all_nodes, params)

        length = state.geodesic_coord[-1]
        print(
            f"[frame {f}]"
            f"  bend={bend:.2f}"
            f"  sigma2={state.sigma2:.6f}"
            f"  length={length:.4f}"
            f"  Y[mid]={state.Y[M // 2]}"
        )

    print("\ndone.")


if __name__ == "__main__":
    main()
