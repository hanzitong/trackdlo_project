# cv2.circle "bad argument / over resolution" エラー

## 症状

`demo_full_pipeline.py` をロボット制御装置で実行すると、`draw_debug()` 内の
`cv2.circle` が "bad argument" または "over resolution" で例外を投げて落ちる。

## 原因

TrackDLO ノードの z 座標が極端に小さい場合（0 に近いが正値）に、
ピンホール投影式が巨大な整数を生成する。

```python
px = int(x3 * FX / z3 + CX)  # z3 が 0.001 程度なら FX/z3 が数十万になる
```

`z3 > 0` チェックは存在するが、ゼロに近い小値を弾けていないため、
`px/py` が画像サイズ（640×480）を大きく超える。`cv2.circle` はこれを拒否する。

## 修正

`px, py` 計算後に画像境界チェックを追加。画像外なら `prev_px = None` で
リセットして描画をスキップ。

```python
h, w = vis_bgr.shape[:2]
# ... (投影計算後)
if not (0 <= px < w and 0 <= py < h):
    prev_px = None   # 画像外ノードをまたぐ接続線も引かない
    continue
```

## 修正ファイル

`src/mz07_demo/demo_full_pipeline.py` の `draw_debug()` 関数
