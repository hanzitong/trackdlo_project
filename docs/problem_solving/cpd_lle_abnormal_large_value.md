# cpd_lle 収束後の異常大座標値バグ

## 症状

`demo_full_pipeline.py` の `initialize_state()` で `cpd_lle` が "converged" を返すが、
出力 Y_fit の X 座標に異常に大きな値（数百〜数千）が含まれる。

```
[cpd_lle] converged=True  sigma2=0.000001
mid node:  X=+842.3120  Y=-0.0031  Z=0.3218  [m]
                 ^^^^^
                 実際は 0.03m 程度のはずなのに数百 m になっている
```

`np.all(np.isfinite(Y_fit))` チェックは通過してしまう（有限値なので）。

## 根本原因の連鎖

### 原因 1: 点群 X に深度復元エラー由来の外れ値が混入

RealSense D405 は背景・反射・エッジ付近で Z が数十〜数百 m になる外れ値を生成することがある。
これらは `images_to_pointcloud()` を通過して点群 X に含まれる。

### 原因 2: Y_init が外れ値に引っ張られて巨大スケールになる

```python
Y_init = np.linspace(X.min(axis=0), X.max(axis=0), NUM_NODES)
```

X に Z=500m の点が 1 つでもあると、Y_init は Z=0 〜 500m にわたる 15 点として生成される。

### 原因 3: C++ 内の X フィルタリングで実ケーブル点群が全除外される

`pure_trackdlo/src/trackdlo.cpp` 内の X フィルタリング処理:

```cpp
if (shortest_dist < 0.1) {   // 最寄りノードから 0.1m 以内の点のみ残す
    X_temp.row(valid_pt_counter) = X_orig.row(i);
    valid_pt_counter += 1;
}
```

Y_init のノードが 0 〜 500m に分布しているとき、実際のケーブル点群（Z ≈ 0.3m）と
最寄りノードの距離が 0.1m を超えるため、ほぼすべての点が除外される。

### 原因 4: N=0 近傍で cpd_lle が "動かないまま収束" を返す

有効点数 N が 0 または極少数の場合:
- sigma2 初期化: `diff_xy.sum() / (D * M * N)` でゼロ除算が発生し 1e-8 にクランプ
- E ステップの P 行列がほぼゼロ
- M ステップで W=0 → Y は動かない
- 収束判定: `pt2pt_dis(Y, Y_0 + G*W) / rows = 0 < tol` → converged=true

Y_init の巨大スケールがそのまま Y_fit として返される。

### 原因 5: isfinite チェックが有限な異常大値を見逃す

```python
if not np.all(np.isfinite(Y_fit)):  # 有限値は通過してしまう
```

`np.isfinite(1e6)` は True を返すため、数百〜数千という有限異常値は検出されない。

## 修正内容

`src/mz07_demo/demo_full_pipeline.py` を 2 箇所変更。

### 変更 1: 上限定数の追加

```python
Z_MAX_M: float = 2.0   # 点群の有効奥行き上限 [m]
```

### 変更 2: `initialize_state()` に Z フィルタと境界チェックを追加

```python
# 外れ値除去: 深度復元エラー由来の Z 異常値が Y_init のスケールを狂わせるため除外する
X = X[X[:, 2] < Z_MAX_M]
if len(X) < NUM_NODES * 3:
    print(f"not enough valid points after Z filter: {len(X)}")
    state.close()
    return None

Y_init = np.linspace(X.min(axis=0), X.max(axis=0), NUM_NODES)
# ... cpd_lle 呼び出し ...

if not np.all(np.isfinite(Y_fit)):
    ...
if np.any(np.abs(Y_fit) > Z_MAX_M * 10):   # 追加: 有限な異常大値を検出
    print(f"  [cpd_lle] abnormally large value detected: max={np.abs(Y_fit).max():.2f}")
    state.close()
    return None
```

## 修正の効果

- Z フィルタにより `X.max()` が正常スケールに収まる
- Y_init が実ケーブル近傍に生成されるようになる
- C++ の 0.1m フィルタが正しく機能し、有効点数 N が確保される
- cpd_lle が実際にフィットするようになる

## 調整が必要なパラメータ

`Z_MAX_M = 2.0` はケーブルがカメラから 2m 以内にある前提。
設置距離が異なる場合はこの値を調整すること。
