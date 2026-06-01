# 共有ライブラリの依存関係と保存場所

pipeline_demo が必要とする `.so` / `.dll` の依存関係と、各ファイルの保存場所をまとめる。

---

## Linux (.so)

### 依存関係

```
libtrackdlo_c.so
├── libstdc++.so.6   (C++ 標準ライブラリ)
├── libm.so.6        (数学ライブラリ)
├── libgcc_s.so.1    (GCC ランタイム)
└── libc.so.6        (C 標準ライブラリ)

libpreprocessing_c.so  ※ PCL / OpenCV は静的リンク済み
├── libstdc++.so.6
├── libm.so.6
├── libgcc_s.so.1
└── libc.so.6
```

上記はすべて Ubuntu 標準パッケージに含まれる。持ち運び先 PC での追加インストールは不要。

### ビルド成果物の保存場所

| ファイル | ビルド元 | 保存先 |
|---|---|---|
| `libtrackdlo_c.so` | `colcon build` → `build/trackdlo_cdll/` | `pipeline_demo/libtrackdlo_c.so` |
| `libpreprocessing_c.so` | スタンドアロンビルド → `build_standalone/preprocessing_cdll/` | `pipeline_demo/libpreprocessing_c.so` |

`libpreprocessing_c.so` は `BUILD_STANDALONE=ON` でビルドした版を使うこと。
通常の `colcon build` 版は PCL / OpenCV の `.so` に動的依存するため持ち運べない。

スタンドアロンビルド手順は `docs/build_guide.md` 参照。

---

## Windows (.dll)

### 依存関係

```
libtrackdlo_c.dll
├── KERNEL32.dll       (Windows 標準)
├── msvcrt.dll         (Windows 標準 C ランタイム)
├── libgcc_s_seh-1.dll (mingw64 GCC ランタイム)
└── libstdc++-6.dll    (mingw64 C++ 標準ライブラリ)

libpreprocessing_c.dll  ※ PCL / OpenCV は静的リンク済み
├── KERNEL32.dll
├── msvcrt.dll
├── libgcc_s_seh-1.dll
├── libstdc++-6.dll
└── libwinpthread-1.dll (mingw64 POSIX スレッドライブラリ)
```

`KERNEL32.dll` / `msvcrt.dll` は Windows 標準なので持ち運び先に常にある。
残り 3 本の mingw64 ランタイム DLL は持ち運し先に存在しないため、同梱が必要。

### ビルド成果物の保存場所

| ファイル | ビルド元 | 保存先 |
|---|---|---|
| `libtrackdlo_c.dll` | `build_win/trackdlo_cdll/` | `pipeline_demo/libtrackdlo_c.dll` |
| `libpreprocessing_c.dll` | `build_win/preprocessing_cdll/` | `pipeline_demo/libpreprocessing_c.dll` |

### mingw64 ランタイム DLL の入手元 (このビルド PC 上)

| DLL | パス |
|---|---|
| `libgcc_s_seh-1.dll` | `/usr/lib/gcc/x86_64-w64-mingw32/10-posix/libgcc_s_seh-1.dll` |
| `libstdc++-6.dll` | `/usr/lib/gcc/x86_64-w64-mingw32/10-posix/libstdc++-6.dll` |
| `libwinpthread-1.dll` | `/usr/x86_64-w64-mingw32/lib/libwinpthread-1.dll` |

これらは `sudo apt install mingw-w64` でインストールされる。

### pipeline_demo への配置

上記 5 本はすべて `pipeline_demo/` 直下にコピー済み。

```
pipeline_demo/
├── libtrackdlo_c.dll
├── libpreprocessing_c.dll
├── libgcc_s_seh-1.dll
├── libstdc++-6.dll
└── libwinpthread-1.dll
```

---

## 依存確認コマンド

```bash
# Linux: 動的依存ライブラリを確認
ldd pipeline_demo/libtrackdlo_c.so
ldd pipeline_demo/libpreprocessing_c.so

# Windows DLL: 依存 DLL を確認 (Ubuntu 上でクロスツールを使う)
x86_64-w64-mingw32-objdump -p pipeline_demo/libtrackdlo_c.dll    | grep "DLL Name"
x86_64-w64-mingw32-objdump -p pipeline_demo/libpreprocessing_c.dll | grep "DLL Name"
```
