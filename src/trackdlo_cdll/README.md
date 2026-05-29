# trackdlo_cdll

`pure_trackdlo`（C++）を C の ABI で再公開するブリッジライブラリ。

## 目的

Python の `ctypes` は C の ABI に準拠した共有ライブラリしか読めない。
C の ABI とは「`double*`・`int` などの単純な型だけを使い、関数名をそのままシンボルとして公開する」という約束事。

`pure_trackdlo` は C++ で書かれているため、そのままでは 2 つの問題がある。

- **name mangling**: C++ コンパイラは `cpd_lle` を `_ZN8trackdlo7cpd_lleE...` のような内部名に変換するため、Python が関数を見つけられない
- **C++ 型**: `Eigen::MatrixXd` や `std::vector` は Python が理解できない型

`trackdlo_cdll` はこの問題を解決するために `extern "C"` で name mangling を無効化し、
`Eigen::MatrixXd` を `double*`（生配列）に変換してから公開する。
Python はその `double*` を `numpy.ndarray` として受け取る。

## 構成

```
trackdlo_cdll/
├── include/trackdlo_c_api.h   C の ABI で公開する関数宣言 (extern "C")
├── src/trackdlo_c_api.cpp     pure_trackdlo の C++ 型 ↔ double* 変換と呼び出し
└── python/trackdlo_cdll.py    ctypes ラッパー (double* ↔ numpy.ndarray 変換)
```

## 使い方

```python
from trackdlo_cdll import TrackdloState, default_params, tracking_step, cpd_lle

state = TrackdloState(num_nodes=15)
params = default_params()
state = tracking_step(state, X, visible_nodes, visible_nodes, params)
```

詳細は `../trackdlo_examples/python/` のサンプルを参照。
