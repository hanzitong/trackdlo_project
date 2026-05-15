# MemoryError: std::bad_alloc in tracking_step

## 症状

```
Iteration until convergence: 2
All nodes visible
MemoryError: std::bad_alloc
```

`verify_trackdlo.py` を実行すると、2フレーム目の `tdlo.tracking_step()` で必ずクラッシュ。

---

## 原因究明

### 連鎖的な障害

```
①  cpd_lle で Np→0 → sigma2 = 0/0 = NaN
②  Y = Y_0 + G*W = NaN  (guide_nodes が NaN に汚染)
③  traverse_euclidean(NaN guide_nodes) → 1個のpriorしか返せない (M=15未満)
④  correspondence_priors ビルドループで priors_vec_2[負インデックス] にアクセス
⑤  ヒープ破壊 → 次の Eigen 行列確保で std::bad_alloc
```

### ① sigma2 が NaN になる理由

`cpd_lle` の M ステップで sigma2 を更新する式:

```cpp
sigma2 = (trXtdPt1X - 2*trPXtT + trTtdP1T) / (Np * D);
```

`Np = P1.sum()` (対応確率の総和)。ノードが点群から大きく離れていると
P の全要素が exp(非常に大きな負値) ≈ 0 → Np ≈ 0 → sigma2 = 0/0 = NaN。

### ③ traverse_euclidean が 1 個しか返せない理由

```cpp
double look_ahead_dist = fabs(geodesic_coord[seg_dist_it+1] - geodesic_coord[seg_dist_it]);
```

`guide_nodes` が NaN なら `line_sphere_intersection` で `delta = NaN`。
`delta < 0` は NaN なので false, `delta > 0` も false → `else` で d1 = NaN。
`isBetween(NaN_pt, ...)` は false → 交点なし → while ループ即時終了。
返り値は初期の 1 node_pair のみ。

### ④ 負インデックスアクセス

`priors_vec_1.size() = 1, priors_vec_2.size() = 1` のとき、
i=1 で:

```cpp
int offset = i - (state.Y.rows() - priors_vec_2.size()); // = 1 - (15-1) = -13
priors_vec_2[offset]  // priors_vec_2[-13] → UB (ヒープ破壊)
```

### 副次的原因: `reg()` が Y_init を無視する

Python 側の `init_state` では:

```python
Y_init = np.linspace(X.min(axis=0), X.max(axis=0), NUM_NODES)
tdlo.reg(X, Y_init, 0.0, M=NUM_NODES)
```

と渡していたが、`utils.cpp` の `reg()` は Y_init を無視して
**常に原点付近 (0, 0..0.1, 0) から初期化**する。
点群が 0.5m 先にある場合、ノードが初期状態で大きくずれており
50 イテレーション後もノードが一点に収束しやすい。
→ `geodesic_coord` が全ゼロ → `look_ahead_dist = 0` → traverse_euclidean 即終了。

---

## 修正箇所

### 1. `cpd_lle` — sigma2 の NaN 防止 (`src/pure_trackdlo/src/trackdlo.cpp`)

```cpp
// sigma2 が NaN になると Y が NaN に汚染される。最小値にクランプする。
if (std::isnan(sigma2) || sigma2 <= 0) sigma2 = 1e-8;
```

### 2. `tracking_step` — correspondence_priors ビルドのバウンドチェック (`trackdlo.cpp`)

`priors_vec_1` / `priors_vec_2` のどちらかが M 個未満なら
`correspondence_priors = {}` にフォールバック (cpd_lle が prior なしで動作)。

```cpp
if (priors_vec_1.size() == M && priors_vec_2.size() == M) {
    // ... 既存ループ (offset >= 0 チェック追加)
}
// 不足の場合は correspondence_priors = {} のまま
```

### 3. Python `init_state` — `reg()` → `cpd_lle()` (`verify_trackdlo.py`, `run_trackdlo.py`)

```python
# reg() は Y_init を無視するため cpd_lle に変更
Y_fit, sig2, _ = tdlo.cpd_lle(X_cont, Y_init, 0.0,
                               beta=5.0, lambda_=1.0, lle_weight=1.0, mu=0.05)
```

LLE 制約によりノード間隔が維持される。

### 4. `VOXEL_M` 拡大 (`verify_trackdlo.py`)

`0.005` → `0.010` m に変更。処理負荷軽減。

### 5. 4 画面常時表示 (`verify_trackdlo.py`)

点数不足でも全4パネルを組み立てて表示するよう変更。

---

## 検証

修正後、`verify_trackdlo.py` を 25 秒間実行。

```
Iteration until convergence: 1
All nodes visible
Iteration until convergence: 1
All nodes visible
...  (クラッシュなし)
```

cpd_lle が 1 イテレーションで収束し続け、正常にトラッキングが継続。
