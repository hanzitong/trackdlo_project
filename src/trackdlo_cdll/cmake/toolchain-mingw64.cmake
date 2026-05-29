# toolchain-mingw64.cmake
#
# Ubuntu 上で Windows 64-bit 向けにクロスコンパイルするためのツールチェーンファイル。
#
# 前提:
#   sudo apt install mingw-w64
#
# 使い方:
#   cmake <source_dir> -DCMAKE_TOOLCHAIN_FILE=<path>/toolchain-mingw64.cmake
#
# CMAKE_TOOLCHAIN_FILE は cmake コマンドに渡すもので、
# CMakeLists.txt の中に書くものではない。

# ターゲット OS とアーキテクチャを指定する
# CMake はこれを見て .dll / .exe などの拡張子を自動で決める
set(CMAKE_SYSTEM_NAME Windows)
set(CMAKE_SYSTEM_PROCESSOR x86_64)

# mingw-w64 クロスコンパイラを指定する
set(CMAKE_C_COMPILER   x86_64-w64-mingw32-gcc)
set(CMAKE_CXX_COMPILER x86_64-w64-mingw32-g++)
set(CMAKE_RC_COMPILER  x86_64-w64-mingw32-windres)

# Windows 向けのライブラリ・ヘッダーの検索ルートを指定する
set(CMAKE_FIND_ROOT_PATH /usr/x86_64-w64-mingw32)

# プログラム (cmake や make 自体) はホスト (Linux) のものを使う
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)

# ライブラリはターゲット (Windows) のものを使う
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)

# ヘッダーはターゲットとホスト両方から探す
# Eigen3 はヘッダーオンリーなのでホスト側 (/usr/include/eigen3) を使えるようにする
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE BOTH)
