"""
trackdlo_wrapper.py

src/trackdlo_cdll/python/trackdlo_cdll.py の「同一ディレクトリ版」。

trackdlo_cdll.py との唯一の違い:
  _load_lib() が build/ や install/ を探すのではなく、
  このファイルと同じディレクトリにある libtrackdlo_c.so を読み込む。
  → setup.sh でコピーした .so を使うため。

それ以外の型定義・関数シグネチャ・クラス定義は trackdlo_cdll.py と同じ内容。
"""

import ctypes
import numpy as np
from pathlib import Path

# このファイルと同じディレクトリにある libtrackdlo_c.so を読み込む
_HERE = Path(__file__).resolve().parent
_SO   = _HERE / "libtrackdlo_c.so"

if not _SO.exists():
    raise FileNotFoundError(
        f"{_SO} が見つかりません。\n"
        "  先に setup.sh を実行してください:\n"
        "    bash setup.sh"
    )

_lib = ctypes.CDLL(str(_SO))


# ================================================================
# TdloParams: C 構造体に対応する ctypes.Structure
#
# _fields_ の順番は trackdlo_c_api.h の TdloParams 定義と一致させる。
# ctypes はメモリ位置を名前ではなく順番で決めるため、順序が重要。
# ================================================================
class TdloParams(ctypes.Structure):
    _fields_ = [
        ("beta",                 ctypes.c_double),
        ("lambda_",              ctypes.c_double),   # C 側フィールド名は lambda
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
# argtypes/restype を登録しないと ctypes はデフォルトで int を返すと仮定する。
# double や void* を返す関数では必ず登録すること。
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
# TrackdloState: C++ オブジェクトを void* ハンドルでラップするクラス
#
# C++ 側で new したメモリを Python 側で解放するため、
# with 文 (推奨) または close() を使う。
# ================================================================
class TrackdloState:
    def __init__(self, num_nodes: int):
        self._handle = _lib.tdlo_state_create(num_nodes)
        if not self._handle:
            raise RuntimeError("tdlo_state_create が NULL を返しました")

    def close(self) -> None:
        if self._handle:
            _lib.tdlo_state_free(self._handle)
            self._handle = None

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    @property
    def Y(self) -> np.ndarray:
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
            self._handle, buf.ctypes.data_as(_dp), ctypes.byref(n)
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
        raise AttributeError("guide_nodes は write-only です")

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
    return _lib.tdlo_default_params()


def tracking_step(
    state: TrackdloState,
    X: np.ndarray,
    visible_nodes,
    visible_nodes_extended,
    params: TdloParams,
) -> None:
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
    """Returns: (Y_updated, sigma2_updated, converged)"""
    X    = np.ascontiguousarray(X, dtype=np.float64)
    Y    = np.ascontiguousarray(Y, dtype=np.float64)
    sig2 = ctypes.c_double(sigma2)

    if visible_nodes is None:
        vn_ptr = _ip()
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
    Y   = np.ascontiguousarray(Y, dtype=np.float64)
    out = np.empty_like(Y)
    _lib.tdlo_sort_pts(Y.ctypes.data_as(_dp), len(Y), out.ctypes.data_as(_dp))
    return out
