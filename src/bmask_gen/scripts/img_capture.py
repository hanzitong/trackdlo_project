# =============================================================================
# img_capture.py
#
# カメラ映像を見ながら 's' キーで画像を保存するデータ収集スクリプト。
# アノテーション前の生 PNG を data/captured/ に蓄積する。
#
# 使い方:
#   python3 scripts/img_capture.py
#   → ウィンドウが開く
#   → 's' キーで data/captured/img_XXXX.png を保存
#   → ESC で終了
# =============================================================================

from pathlib import Path
import cv2

# スクリプトの 2 階層上 = bmask_gen/ ルート
ROOT = Path(__file__).resolve().parent.parent

# 保存先ディレクトリを作成する
# parents=True: 中間ディレクトリも含めて再帰的に作成 (mkdir -p と同じ)
# exist_ok=True: すでに存在する場合もエラーにしない
save_dir = ROOT / "data/captured"
save_dir.mkdir(parents=True, exist_ok=True)

# カメラを開く (0 = /dev/video0, CAP_V4L2 = Linux V4L2 バックエンド)
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
if not cap.isOpened():
    raise RuntimeError("cannot open camera !")

# 解像度設定 (学習時の入力サイズ 640x480 に合わせる)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

count = 0   # 保存枚数のカウンタ

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow("camera", frame)

    # cv2.waitKey(1): 1ms 待機してキー入力を取得する
    # & 0xFF: 上位バイトを除去して ASCII コードを取り出す
    key = cv2.waitKey(1) & 0xFF

    # 's' キーで現在のフレームを保存
    # ord('s'): 文字 's' の ASCII コード (115) を取得する
    if key == ord('s'):
        # f"img_{count:04d}.png": count を 4 桁ゼロパディングでフォーマット
        #   例: count=0 → "img_0000.png", count=12 → "img_0012.png"
        filename = save_dir / f"img_{count:04d}.png"
        cv2.imwrite(str(filename), frame)
        print("saved:", filename)
        count += 1

    # 27 = ESC キーのコード
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
