# RealSense D405 depth 単位ずれによる座標値 10 倍誤差

## 症状

`infer_trackdlo.py` および `test_cablekeypoint_pipeline.py` を実行すると、
TrackDLO が出力するケーブルキーポイントの Z 座標が実際の距離の約 10 倍になっていた。

例: カメラからケーブルまでの実距離が約 30cm のとき

```
[frame 0001] mid node:  X=+0.0312  Y=-0.0091  Z=3.2180  [m]
                                              ^^^^^^^^^^
                                              実際は 0.322m のはずなのに 3.2m になっている
```

また、X・Y 座標の範囲も実際のケーブルの大きさに対して 10 倍程度広くなっていた。

## 原因の連鎖

### 原因 1: take_depth_rgb_image.py が raw 値をそのまま保存していた

```python
# take_depth_rgb_image.py (修正前)
np_depth_frame: np.ndarray = np.asanyarray(rs_depth_frame.get_data())
cv2.imwrite(str(depth_path), np_depth_frame)   # raw センサー値をそのまま保存
```

`rs_depth_frame.get_data()` が返す uint16 の値は、カメラの depth scale によって
実際の距離に変換する必要がある。保存時に変換しなかったため、PNG には
「0.1mm 単位の整数」がそのまま記録されていた。

### 原因 2: C++ preprocessing コードが mm 単位を前提としていた

```cpp
// preprocessing/src/preprocessing.cpp
double pc_z = depth.at<uint16_t>(i, j) / 1000.0;
//                                        ^^^^^^
//           uint16 値が mm 単位であると決め打ちして /1000 で m 変換
```

このコードは depth 値が mm 単位 (1mm/unit = 0.001 m/unit) であることを前提にしている。

### 原因 3: D405 の depth scale は 0.0001 m/unit (0.1mm 単位)

RealSense D405 は近距離精度のために、デフォルトの depth scale が
一般的な D415/D435 (0.001 m/unit = 1mm/unit) とは異なり、
**0.0001 m/unit (= 0.1mm/unit)** に設定されている場合がある。

```python
# カメラから実際の depth scale を取得
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
# D405 では 0.0001 が返る
```

このため:
- raw 値 3218 は 3218 × 0.0001 m = 0.3218m を意味する (正しい距離)
- しかし C++ は 3218 / 1000.0 = 3.218m と計算してしまう (10 倍ズレ)

### ズレの大きさの確認

sample_data のケーブル部分の depth PNG 値を実測して確認した:

| フレーム | raw 値 (mean) | /1000 = m | /10000 = m | 推定実距離 |
|---|---|---|---|---|
| depth_0001.png | 3218 | 3.218m | 0.322m | 約 30cm |
| depth_0002.png | 3199 | 3.199m | 0.320m | 約 30cm |
| depth_0003.png | 3229 | 3.229m | 0.323m | 約 30cm |

`/10000` の結果 (約 0.32m) がユーザーの実測値 (約 30cm) と一致することで
depth scale = 0.0001 m/unit であることが確定した。

### 変数名が原因の見落としを誘発

```python
# infer_trackdlo.py (修正前)
depth_mm: np.ndarray = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
#  ^^^^^^^^^^
#  変数名が depth_mm なので mm 単位に見えるが、実体は raw センサー値 (0.1mm 単位)
```

変数名 `depth_mm` が「mm 単位の値が入っている」と誤解させる命名になっていたため、
問題の発見が遅れた。

## 修正方針

**C++ 側の根本修正**は preprocessing.cpp の `/1000.0` を `depth_scale` 引数対応に
変更することだが、C++ の再コンパイルが必要になる。

今回は **Python 側で mm 換算してから C++ に渡す応急処置**を採用した。

```python
# Python 側での変換 (各スクリプト共通)
# raw × depth_scale × 1000 = mm 換算値
# 例: 3218 × 0.0001 × 1000 = 321.8 mm → uint16(321) → C++ で 321/1000 = 0.321m
depth_mm: np.ndarray = np.clip(
    depth_raw.astype(np.float32) * DEPTH_SCALE * 1000.0, 0, 65535
).astype(np.uint16)
```

C++ 側への TODO コメントも追記済み:

```cpp
// preprocessing/src/preprocessing.cpp
// TODO(根本修正): ここでは depth が uint16 mm 単位 (1mm/unit) であることを前提に
// /1000.0 で m 変換している。RealSense D405 は depth_scale = 0.0001 m/unit
// (0.1mm/unit) を使うため、本来は /10000.0 にするか、depth_scale を引数として
// 受け取るよう API を変更すべき。現状は呼び出し側 (Python) で raw 値を mm 換算して
// から渡す応急処置で対応している。
double pc_z = depth.at<uint16_t>(i, j) / 1000.0;
```

## 修正後の各スクリプトでの depth の流れ

### take_depth_rgb_image.py (キャプチャ)

```python
# 修正後: 保存前に mm 換算する
depth_raw: np.ndarray = np.asanyarray(depth_frame.get_data())   # raw uint16
depth_mm: np.ndarray = np.clip(
    depth_raw.astype(np.float32) * DEPTH_SCALE * 1000.0, 0, 65535
).astype(np.uint16)
cv2.imwrite(str(depth_path), depth_mm)   # mm 単位で保存
```

### infer_trackdlo.py (オフライン推論)

```python
# 修正後: load_frames() 内で変換
depth_raw: np.ndarray = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
depth_mm: np.ndarray = np.clip(
    depth_raw.astype(np.float32) * DEPTH_SCALE * 1000.0, 0, 65535
).astype(np.uint16)
# 以降 depth_mm を使用 (mm 単位で統一)
```

注意: 修正前の sample_data PNG は raw 値で保存されているため、
`DEPTH_SCALE = 0.0001` のまま使用する必要がある。
修正後の `take_depth_rgb_image.py` で新たにキャプチャしたデータは
すでに mm 単位のため `DEPTH_SCALE = 1.0` (変換不要) に変更すること。

### test_cablekeypoint_pipeline.py (ライブパイプライン)

```python
# 修正後: カメラ取得直後に変換
DEPTH_SCALE: float = _depth_sensor.get_depth_scale()   # 実機から取得 (0.0001)

# ループ内
depth_raw: np.ndarray = np.asanyarray(depth_frame.get_data())
depth_mm: np.ndarray = np.clip(
    depth_raw.astype(np.float32) * DEPTH_SCALE * 1000.0, 0, 65535
).astype(np.uint16)
# 以降 depth_mm を使用
```

## 単位統一の設計方針

mz07_demo 内の全スクリプトは **depth = uint16, mm 単位** で統一されている。

```
RealSense raw (0.1mm 単位)
    │
    ▼  × DEPTH_SCALE × 1000  (各スクリプトの入口で一度だけ変換)
depth_mm (uint16, mm 単位)
    │
    ▼  C++ images_to_pointcloud() が内部で /1000.0
3D 点群 X (float64, m 単位)
    │
    ▼  TrackDLO tracking_step()
ノード座標 Y (float64, m 単位)
```

変換は必ずデータが入ってくる最初の箇所 (ファイル読み込みまたはカメラ取得直後) で
行い、それ以降のパイプラインは常に mm 換算済みの値を使う。
