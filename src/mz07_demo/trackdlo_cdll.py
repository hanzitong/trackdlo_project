"""
trackdlo_cdll: ctypes による pure_trackdlo の Python ラッパー

ctypes は Python 標準ライブラリ。pip install 不要。
pybind11 と異なり Python 側にコンパイルステップがない (.py ファイルだけ)。
ただし型変換を手動で行う必要がある。

[使い方]
    from trackdlo_cdll.python.trackdlo_cdll import TrackdloState, default_params, tracking_step

[前提]
    colcon build --packages-up-to trackdlo_cdll を実行して
    libtrackdlo_c.so を生成しておくこと。

[.so のパスを上書きしたい場合]
    import os
    os.environ["TRACKDLO_LIB_PATH"] = "/path/to/libtrackdlo_c.so"
    import trackdlo_cdll   # ← 環境変数をセットしてから import する

    pipeline_demo のように .so をスクリプトと同じディレクトリに置く場合に使う。
    環境変数は import より前にセットする必要がある (_load_lib は import 時に実行されるため)。
"""

import ctypes
import numpy as np
import os
from pathlib import Path


# ================================================================
# ライブラリのロード
#
# 優先順位:
#   1. 環境変数 TRACKDLO_LIB_PATH が指すパス (明示的な上書き)
#   2. colcon build ディレクトリ
#   3. colcon install ディレクトリ
# ================================================================

def _load_lib() -> ctypes.CDLL:
    # 環境変数による上書き (pipeline_demo など .so を別の場所に置く場合に使う)
    env_path = os.environ.get("TRACKDLO_LIB_PATH")
    if env_path:
        p = Path(env_path)
        if not p.exists():
            raise FileNotFoundError(f"TRACKDLO_LIB_PATH points to a missing file: {p}")
        return ctypes.CDLL(str(p))

    here    = Path(__file__).resolve()
    ws_root = here.parents[3]  # src/trackdlo_cdll/python/ から3つ上がワークスペースルート
    candidates = [
        ws_root / "build"   / "trackdlo_cdll" / "libtrackdlo_c.so",
        ws_root / "install" / "trackdlo_cdll" / "lib" / "libtrackdlo_c.so",
    ]
    for p in candidates:
        if p.exists():
            return ctypes.CDLL(str(p))
    raise FileNotFoundError(
        "libtrackdlo_c.so not found.\n"
        "  Run: colcon build --packages-up-to trackdlo_cdll\n"
        "Searched paths:\n" + "\n".join(f"  {p}" for p in candidates)
    )


_lib = _load_lib()


# ================================================================
# TdloParams: C 構造体に対応する ctypes.Structure
#
# ctypes.Structure は C の struct と同じメモリレイアウトを持つクラス。
# _fields_ でフィールド名と型を (名前, ctypes型) のリストで定義する。
# 順番が C 側の定義と一致していなければならない。
#
# [注意] C 側のフィールド名 lambda は Python の予約語なので、
#        Python 側では lambda_ という名前にしている。
#        ctypes はメモリ位置を名前ではなく順番で決めるため問題ない。
# ================================================================
class TdloParams(ctypes.Structure):
    _fields_ = [
        ("beta",                 ctypes.c_double),
        ("lambda_",              ctypes.c_double),   # C 側の lambda フィールド
        ("alpha",                ctypes.c_double),
        ("k_vis",                ctypes.c_double),
        ("mu",                   ctypes.c_double),
        ("max_iter",             ctypes.c_int),
        ("tol",                  ctypes.c_double),
        ("beta_pre_proc",        ctypes.c_double),
        ("lambda_pre_proc",      ctypes.c_double),
        ("lle_weight",           ctypes.c_double),
        ("visibility_threshold", ctypes.c_double),
    ]


# ================================================================
# 関数シグネチャの登録
#
# argtypes と restype を登録しないと ctypes はデフォルトで int を想定する。
# double や ポインタを返す関数では必ず登録すること (サイレントバグの原因になる)。
#
# ctypes の型対応:
#   c_double  ↔  double
#   c_int     ↔  int
#   c_void_p  ↔  void* (不透明ハンドル)
#   POINTER(c_double) ↔  double*
# ================================================================
_dp = ctypes.POINTER(ctypes.c_double)
_ip = ctypes.POINTER(ctypes.c_int)

_lib.tdlo_state_create.restype  = ctypes.c_void_p
_lib.tdlo_state_create.argtypes = [ctypes.c_int]

_lib.tdlo_state_free.restype    = None
_lib.tdlo_state_free.argtypes   = [ctypes.c_void_p]

_lib.tdlo_state_num_nodes.restype  = ctypes.c_int
_lib.tdlo_state_num_nodes.argtypes = [ctypes.c_void_p]

_lib.tdlo_state_get_Y.restype  = None
_lib.tdlo_state_get_Y.argtypes = [ctypes.c_void_p, _dp]

_lib.tdlo_state_set_Y.restype  = None
_lib.tdlo_state_set_Y.argtypes = [ctypes.c_void_p, _dp, ctypes.c_int]

_lib.tdlo_state_get_sigma2.restype  = ctypes.c_double
_lib.tdlo_state_get_sigma2.argtypes = [ctypes.c_void_p]

_lib.tdlo_state_set_sigma2.restype  = None
_lib.tdlo_state_set_sigma2.argtypes = [ctypes.c_void_p, ctypes.c_double]

_lib.tdlo_state_set_geodesic_coord.restype  = None
_lib.tdlo_state_set_geodesic_coord.argtypes = [ctypes.c_void_p, _dp, ctypes.c_int]

_lib.tdlo_state_get_geodesic_coord.restype  = None
_lib.tdlo_state_get_geodesic_coord.argtypes = [ctypes.c_void_p, _dp, _ip]

_lib.tdlo_state_set_guide_nodes.restype  = None
_lib.tdlo_state_set_guide_nodes.argtypes = [ctypes.c_void_p, _dp, ctypes.c_int]

_lib.tdlo_default_params.restype  = TdloParams
_lib.tdlo_default_params.argtypes = []

_lib.tdlo_tracking_step.restype  = None
_lib.tdlo_tracking_step.argtypes = [
    ctypes.c_void_p,
    _dp, ctypes.c_int,
    _ip, ctypes.c_int,
    _ip, ctypes.c_int,
    ctypes.POINTER(TdloParams),
]

_lib.tdlo_cpd_lle.restype  = ctypes.c_int
_lib.tdlo_cpd_lle.argtypes = [
    _dp, ctypes.c_int,
    _dp, ctypes.c_int,
    ctypes.POINTER(ctypes.c_double),
    ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double,
    ctypes.c_int, ctypes.c_double, ctypes.c_int,
    ctypes.c_double,
    _ip, ctypes.c_int,
    ctypes.c_double, ctypes.c_double,
]

_lib.tdlo_sort_pts.restype  = None
_lib.tdlo_sort_pts.argtypes = [_dp, ctypes.c_int, _dp]

_lib.tdlo_reg.restype  = None
_lib.tdlo_reg.argtypes = [
    _dp, ctypes.c_int,
    _dp, ctypes.c_int,
    ctypes.POINTER(ctypes.c_double),
    ctypes.c_double, ctypes.c_int,
]


# ================================================================
# TrackdloState: 不透明ハンドルをラップする Python クラス
#
# C++ の TrackdloState オブジェクトを _handle (void*) として保持する。
#
# メモリ管理の方針:
#   C++ 側で new したオブジェクトを Python 側で確実に delete するために
#   3つの手段を用意している。
#
#   1. with 文 (推奨)
#      __enter__ / __exit__ でコンテキストマネージャとして使う。
#      with ブロックを抜けると例外の有無に関わらず必ず close() が呼ばれる。
#      C++ の RAII に相当する Python のイディオム。
#
#   2. close() の明示的呼び出し
#      try / finally と組み合わせて使う。
#      with 文が使えない場面 (クラスのメンバ変数として持つ場合など) に使う。
#
#   3. __del__ (フォールバック)
#      上記2つを忘れたときの最後の砦。
#      ただし Python の __del__ は以下の場合に呼ばれないことがある:
#        - 循環参照がある場合 (CPython の循環 GC は __del__ を持つ
#          オブジェクトを即座に回収しない)
#        - インタープリタ終了時
#      そのため __del__ だけに頼るのは危険。
# ================================================================
class TrackdloState:
    def __init__(self, num_nodes: int):
        self._handle = _lib.tdlo_state_create(num_nodes)
        if not self._handle:
            raise RuntimeError("tdlo_state_create returned NULL (possibly out of memory)")

    def close(self) -> None:
        """C++ 側のメモリを明示的に解放する。二重解放は安全に無視される。"""
        if self._handle:
            _lib.tdlo_state_free(self._handle)
            self._handle = None

    def __del__(self):
        # close() を呼び忘れたときのフォールバック。
        # with 文や明示的な close() を使っていれば _handle は既に None。
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # with ブロックを抜けるとき (例外の有無に関わらず) に呼ばれる
        self.close()
        return False  # 例外を握りつぶさない

    @property
    def Y(self) -> np.ndarray:
        """ノード座標を (M, 3) の numpy 配列として返す"""
        M   = _lib.tdlo_state_num_nodes(self._handle)
        out = np.empty((M, 3), dtype=np.float64)
        _lib.tdlo_state_get_Y(self._handle, out.ctypes.data_as(_dp))
        return out

    @Y.setter
    def Y(self, arr: np.ndarray):
        arr = np.ascontiguousarray(arr, dtype=np.float64)
        _lib.tdlo_state_set_Y(self._handle, arr.ctypes.data_as(_dp), arr.shape[0])

    @property
    def sigma2(self) -> float:
        return _lib.tdlo_state_get_sigma2(self._handle)

    @sigma2.setter
    def sigma2(self, v: float):
        _lib.tdlo_state_set_sigma2(self._handle, float(v))

    @property
    def geodesic_coord(self) -> np.ndarray:
        M   = _lib.tdlo_state_num_nodes(self._handle)
        buf = np.empty(M, dtype=np.float64)
        n   = ctypes.c_int(0)
        _lib.tdlo_state_get_geodesic_coord(
            self._handle,
            buf.ctypes.data_as(_dp),
            ctypes.byref(n),
        )
        return buf[:n.value]

    @geodesic_coord.setter
    def geodesic_coord(self, arr):
        arr = np.ascontiguousarray(arr, dtype=np.float64)
        _lib.tdlo_state_set_geodesic_coord(
            self._handle, arr.ctypes.data_as(_dp), len(arr)
        )

    @property
    def guide_nodes(self):
        raise AttributeError("guide_nodes is write-only")

    @guide_nodes.setter
    def guide_nodes(self, arr: np.ndarray):
        arr = np.ascontiguousarray(arr, dtype=np.float64)
        _lib.tdlo_state_set_guide_nodes(
            self._handle, arr.ctypes.data_as(_dp), arr.shape[0]
        )


# ================================================================
# 公開 API
# ================================================================

def default_params() -> TdloParams:
    """デフォルト値で初期化された TdloParams を返す"""
    return _lib.tdlo_default_params()


def tracking_step(
    state: TrackdloState,
    X: np.ndarray,
    visible_nodes,
    visible_nodes_extended,
    params: TdloParams,
) -> None:
    """
    1 フレームのトラッキング処理。state.Y を in-place で更新する。

    X:                      点群 (N, 3) float64
    visible_nodes:          可視ノードのインデックスリスト
    visible_nodes_extended: 拡張可視ノードのインデックスリスト
    params:                 TdloParams

    [注意] np.ascontiguousarray で作った配列のポインタを渡す。
    配列の参照が呼び出し中に GC されないよう、ローカル変数として保持する。
    """
    X   = np.ascontiguousarray(X,                     dtype=np.float64)
    vn  = np.ascontiguousarray(visible_nodes,          dtype=np.int32)
    vne = np.ascontiguousarray(visible_nodes_extended, dtype=np.int32)

    _lib.tdlo_tracking_step(
        state._handle,
        X.ctypes.data_as(_dp),   len(X),
        vn.ctypes.data_as(_ip),  len(vn),
        vne.ctypes.data_as(_ip), len(vne),
        ctypes.byref(params),
    )


def cpd_lle(
    X: np.ndarray,
    Y: np.ndarray,
    sigma2: float,
    beta: float,
    lambda_: float,
    lle_weight: float,
    mu: float,
    max_iter: int = 30,
    tol: float = 0.0001,
    include_lle: bool = True,
    alpha: float = 0.0,
    visible_nodes=None,
    k_vis: float = 0.0,
    visibility_threshold: float = 0.01,
) -> tuple:
    """
    CPD + LLE 登録。

    Returns:
        (Y_updated, sigma2_updated, converged)
    Y_updated は更新後の (M, 3) numpy 配列。
    """
    X    = np.ascontiguousarray(X, dtype=np.float64)
    Y    = np.ascontiguousarray(Y, dtype=np.float64)
    sig2 = ctypes.c_double(sigma2)

    # visible_nodes が None の場合は NULL ポインタを渡す
    if visible_nodes is None:
        vn_ptr = _ip()   # NULL ポインタ
        vn_len = 0
    else:
        vn     = np.ascontiguousarray(visible_nodes, dtype=np.int32)
        vn_ptr = vn.ctypes.data_as(_ip)
        vn_len = len(vn)

    converged = _lib.tdlo_cpd_lle(
        X.ctypes.data_as(_dp),   len(X),
        Y.ctypes.data_as(_dp),   len(Y),
        ctypes.byref(sig2),
        beta, lambda_, lle_weight, mu,
        max_iter, tol, int(include_lle),
        alpha,
        vn_ptr, vn_len,
        k_vis, visibility_threshold,
    )
    return Y, sig2.value, bool(converged)


def sort_pts(Y: np.ndarray) -> np.ndarray:
    """ノードを最近傍順に並べ替えた (M, 3) 配列を返す"""
    Y   = np.ascontiguousarray(Y, dtype=np.float64)
    out = np.empty_like(Y)
    _lib.tdlo_sort_pts(Y.ctypes.data_as(_dp), len(Y), out.ctypes.data_as(_dp))
    return out


def reg(
    pts: np.ndarray,
    Y: np.ndarray,
    sigma2: float,
    M: int,
    mu: float = 0.0,
    max_iter: int = 50,
) -> tuple:
    """
    簡易 CPD 登録 (初期化用)。

    Returns:
        (Y_updated, sigma2_updated)
    """
    pts  = np.ascontiguousarray(pts, dtype=np.float64)
    Y    = np.ascontiguousarray(Y,   dtype=np.float64)
    sig2 = ctypes.c_double(sigma2)

    _lib.tdlo_reg(
        pts.ctypes.data_as(_dp), len(pts),
        Y.ctypes.data_as(_dp),   M,
        ctypes.byref(sig2),
        mu, max_iter,
    )
    return Y, sig2.value
