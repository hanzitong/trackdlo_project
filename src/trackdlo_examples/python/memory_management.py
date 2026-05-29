"""
memory_management.py

TrackdloState のメモリ管理パターンを示すサンプル。

C++ の new で確保したオブジェクトを Python 側で確実に delete するための
3つのパターンを示す。RealSense 不要。合成点群で動作する。

実行:
    python3 src/trackdlo_examples/python/memory_management.py
"""

import sys
import math
import numpy as np
from pathlib import Path

_SRC = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_SRC / "trackdlo_cdll" / "python"))

from trackdlo_cdll import TrackdloState, default_params, tracking_step, cpd_lle, sort_pts


def make_cable_cloud(N: int, bend: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t   = np.linspace(0.0, 1.0, N)
    pts = np.column_stack([
        t + rng.normal(0, 0.005, N),
        bend * np.sin(t * math.pi) + rng.normal(0, 0.005, N),
        np.ones(N) + rng.normal(0, 0.005, N),
    ])
    return pts.astype(np.float64)


def initialize(X: np.ndarray, M: int) -> TrackdloState:
    """TrackdloState を作って返す。呼び出し側がメモリ管理の責任を持つ。"""
    state          = TrackdloState(M)
    Y_init         = np.linspace(X.min(axis=0), X.max(axis=0), M)
    Y_fit, sig2, _ = cpd_lle(X, Y_init, 0.0,
                              beta=5.0, lambda_=1.0, lle_weight=1.0, mu=0.05)
    state.Y             = sort_pts(Y_fit)
    state.sigma2        = sig2
    dists               = np.linalg.norm(np.diff(state.Y, axis=0), axis=1)
    state.geodesic_coord = np.concatenate([[0.0], np.cumsum(dists)])
    state.guide_nodes   = state.Y
    return state


M      = 10
X      = make_cable_cloud(200, bend=0.0, seed=0)
params = default_params()
params.max_iter = 150
params.tol      = 1e-4
all_idx = list(range(M))

print("=" * 50)

# ============================================================
# パターン 1: with 文 (推奨)
#
# with ブロックを抜けるとき、例外が発生しても __exit__ が呼ばれ
# close() → tdlo_state_free() が確実に実行される。
# C++ の RAII (スコープを抜けたらデストラクタ) と同じ考え方。
# ============================================================
print("[Pattern 1] with 文")

with initialize(X, M) as state:
    tracking_step(state, make_cable_cloud(200, 0.05, 1), all_idx, all_idx, params)
    print(f"  node 0: {state.Y[0]}")
    print(f"  node {M-1}: {state.Y[M-1]}")
# ← ここで with を抜けた瞬間に close() が呼ばれ C++ メモリが解放される

print("  → with ブロック終了時に解放済み\n")


# ============================================================
# パターン 2: try / finally
#
# with 文が使えない場面 (クラスのメンバ変数として state を持つ場合など)
# では try / finally で明示的に close() を呼ぶ。
# ============================================================
print("[Pattern 2] try / finally")

state = initialize(X, M)
try:
    tracking_step(state, make_cable_cloud(200, 0.10, 2), all_idx, all_idx, params)
    print(f"  node 0: {state.Y[0]}")
finally:
    state.close()
    print("  → finally で close() 呼び出し済み\n")


# ============================================================
# パターン 3: __del__ に任せる (非推奨・フォールバック)
#
# close() を呼ばなくても GC が __del__ を呼ぶが、
# 循環参照がある場合やインタープリタ終了時には保証されない。
# ============================================================
print("[Pattern 3] __del__ に任せる (フォールバック)")

state = initialize(X, M)
tracking_step(state, make_cable_cloud(200, 0.15, 3), all_idx, all_idx, params)
print(f"  node 0: {state.Y[0]}")
del state
print("  → del 後に __del__ 経由で解放済み\n")

print("done.")
