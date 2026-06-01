# ビルドガイド

このドキュメントでは、`preprocessing` / `preprocessing_cdll` / `trackdlo_cdll` の
Ubuntu 向けビルドと Windows 向けクロスコンパイルの手順を記載する。

---

## Ubuntu 向けビルド（通常の開発フロー）

```bash
# ワークスペースルート (trackdlo_project/) で実行する

# 全パッケージを一括ビルド
colcon build

# preprocessing + preprocessing_cdll だけビルド
colcon build --packages-select preprocessing preprocessing_cdll

# trackdlo_cdll だけビルド
colcon build --packages-select pure_trackdlo trackdlo_cdll
```

ビルド成果物:
- `install/preprocessing/lib/libpreprocessing.a`
- `install/preprocessing_cdll/lib/libpreprocessing_c.so`
- `install/trackdlo_cdll/lib/libtrackdlo_c.so`

---

## Windows 向けクロスコンパイル（Ubuntu から）

### 前提条件

Ubuntu 上で以下がインストール済みであること:

```bash
sudo apt install mingw-w64
```

`/usr/bin/x86_64-w64-mingw32-g++-posix` が存在することを確認する
(win32 版ではなく **posix 版** を使う必要がある。理由: OpenCV が `std::mutex` 等を使うため)。

### ディレクトリ構成

```
trackdlo_project/
├── src/
│   ├── pure_trackdlo/
│   ├── trackdlo_cdll/
│   │   └── cmake/toolchain-mingw64.cmake   ← 共通ツールチェーンファイル
│   ├── preprocessing/
│   └── preprocessing_cdll/
├── build_win/                               ← Windows 向けビルドツリー
│   ├── COLCON_IGNORE                        ← colcon が誤スキャンしないよう配置済み
│   ├── opencv/                              ← OpenCV クロスビルド (初回のみ)
│   ├── pure_trackdlo/
│   ├── preprocessing/
│   └── preprocessing_cdll/
└── install_win/                             ← Windows 向けインストール先
    ├── include/
    ├── lib/
    │   ├── libopencv_core454.a
    │   ├── libopencv_imgproc454.a
    │   ├── libpure_trackdlo.a
    │   ├── libpreprocessing.a
    │   └── opencv4/3rdparty/libzlib.a
    └── lib/cmake/
```

---

### Step 1: OpenCV のクロスビルド（初回のみ）

OpenCV 4.5.4 のソースを取得してクロスビルドする。
一度 `install_win/` にインストールすれば以降は不要。

```bash
# ソース取得
wget https://github.com/opencv/opencv/archive/refs/tags/4.5.4.tar.gz -O /tmp/opencv-4.5.4.tar.gz
tar xzf /tmp/opencv-4.5.4.tar.gz -C /tmp

# ビルド
mkdir -p build_win/opencv && cd build_win/opencv
cmake /tmp/opencv-4.5.4 \
  -DCMAKE_TOOLCHAIN_FILE=../../src/trackdlo_cdll/cmake/toolchain-mingw64.cmake \
  -DCMAKE_INSTALL_PREFIX=../../install_win \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_SHARED_LIBS=OFF \
  -DBUILD_TESTS=OFF \
  -DBUILD_PERF_TESTS=OFF \
  -DBUILD_EXAMPLES=OFF \
  -DBUILD_DOCS=OFF \
  -DBUILD_LIST=core,imgproc \
  -DOPENCV_3P_LIB_INSTALL_PATH=lib/opencv4/3rdparty \
  -DWITH_IPP=OFF \
  -DWITH_TBB=OFF \
  -DWITH_OPENMP=OFF \
  -DWITH_EIGEN=OFF \
  -DWITH_1394=OFF \
  -DWITH_FFMPEG=OFF \
  -DWITH_GSTREAMER=OFF \
  -DWITH_V4L=OFF \
  -DWITH_GTK=OFF \
  -DWITH_QT=OFF \
  -DWITH_WIN32UI=OFF \
  -DWITH_JPEG=OFF \
  -DWITH_PNG=OFF \
  -DWITH_TIFF=OFF \
  -DWITH_WEBP=OFF \
  -DWITH_JASPER=OFF \
  -DWITH_OPENJPEG=OFF \
  -DWITH_OPENEXR=OFF \
  -DWITH_PROTOBUF=OFF \
  -DWITH_QUIRC=OFF \
  -DWITH_ADE=OFF \
  -DBUILD_ZLIB=ON
make -j$(nproc) install
cd ../..
```

`-DWITH_JPEG=OFF` 等を指定している理由:
OpenCV をクロスビルドすると `OpenCVModules.cmake` に使用した 3rdparty ライブラリの
cmake ターゲットが列挙される。不要なライブラリを有効にするとそのターゲットが
cmake find_package 時に「定義されていない」エラーを起こすため、
core + imgproc に不要な画像コーデック系はすべて OFF にしている。

---

### Step 2: pure_trackdlo のクロスビルド（初回または pure_trackdlo 変更時）

```bash
mkdir -p build_win/pure_trackdlo && cd build_win/pure_trackdlo
cmake ../../src/pure_trackdlo \
  -DCMAKE_TOOLCHAIN_FILE=../../src/trackdlo_cdll/cmake/toolchain-mingw64.cmake \
  -DCMAKE_INSTALL_PREFIX=../../install_win
make -j$(nproc) install
cd ../..
```

---

### Step 3: trackdlo_c.dll のビルド

```bash
mkdir -p build_win/trackdlo_cdll && cd build_win/trackdlo_cdll
cmake ../../src/trackdlo_cdll \
  -DBUILD_FOR_WINDOWS=ON \
  -DCMAKE_TOOLCHAIN_FILE=../../src/trackdlo_cdll/cmake/toolchain-mingw64.cmake \
  -DCMAKE_PREFIX_PATH=../../install_win
make -j$(nproc)
cd ../..
# → build_win/trackdlo_cdll/trackdlo_c.dll
```

---

### Step 4: preprocessing のクロスビルド

```bash
mkdir -p build_win/preprocessing && cd build_win/preprocessing
cmake ../../src/preprocessing \
  -DBUILD_FOR_WINDOWS=ON \
  -DCMAKE_TOOLCHAIN_FILE=../../src/trackdlo_cdll/cmake/toolchain-mingw64.cmake \
  -DCMAKE_PREFIX_PATH=../../install_win \
  -DCMAKE_INSTALL_PREFIX=../../install_win
make -j$(nproc) install
cd ../..
```

---

### Step 5: preprocessing_c.dll のビルド

```bash
mkdir -p build_win/preprocessing_cdll && cd build_win/preprocessing_cdll
cmake ../../src/preprocessing_cdll \
  -DBUILD_FOR_WINDOWS=ON \
  -DCMAKE_TOOLCHAIN_FILE=../../src/trackdlo_cdll/cmake/toolchain-mingw64.cmake \
  -DCMAKE_PREFIX_PATH=../../install_win
make -j$(nproc)
cd ../..
# → build_win/preprocessing_cdll/libpreprocessing_c.dll
```

---

### 生成物の確認

```bash
file build_win/trackdlo_cdll/trackdlo_c.dll
# → PE32+ executable (DLL) (console) x86-64, for MS Windows

file build_win/preprocessing_cdll/libpreprocessing_c.dll
# → PE32+ executable (DLL) (console) x86-64, for MS Windows
```

---

### Windows 側での使い方

生成した DLL を Windows マシンにコピーして Python から ctypes でロードする。
`preprocessing_cdll` は OpenCV を静的リンクしているため、
`opencv_core454.dll` 等の OpenCV DLL を別途配布する必要はない。

ランタイムとして mingw64 の posix スレッドライブラリが必要になる場合がある:

```
libwinpthread-1.dll  ← x86_64-w64-mingw32 の posix スレッドモデルに必要
```

このファイルは Ubuntu の `/usr/x86_64-w64-mingw32/lib/libwinpthread-1.dll` にある。
DLL と同じディレクトリに置くか、Windows の PATH に通す。

---

## トラブルシューティング

### `std::recursive_mutex` does not name a type

mingw64 が win32 スレッドモデルになっている。
`toolchain-mingw64.cmake` が `g++-posix` を指定しているか確認する。

### `Some (but not all) targets in this export set were already defined`

`OpenCVModules.cmake` が参照する 3rdparty ターゲットの `.a` ファイルが `install_win/` にない。
Step 1 の cmake コマンドで `-DWITH_JPEG=OFF` 等が正しく指定されているか確認し、
`build_win/opencv/` と `install_win/` の OpenCV 関連ファイルを削除して Step 1 からやり直す。

### colcon build で `build_win/opencv/python_loader` のエラーが出る

`build_win/COLCON_IGNORE` が存在するか確認する:

```bash
ls build_win/COLCON_IGNORE
```

なければ作成する:

```bash
touch build_win/COLCON_IGNORE
```
