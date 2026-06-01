cmake_minimum_required(VERSION 3.14)
project(trackdlo_cdll)

set(CMAKE_CXX_STANDARD 17)
add_compile_options(-O3)

# =============================================================================
# [ビルドの全体像]
#
# 生成物: libtrackdlo_c.so (Linux) / trackdlo_c.dll (Windows)
#
#   trackdlo_c_api.cpp (extern "C" で公開した C ラッパー関数)
#       + libpure_trackdlo.a (アルゴリズム本体、静的リンクで吸収)
#   ↓ コンパイル & リンク
#   libtrackdlo_c.so / trackdlo_c.dll  ← Python が ctypes.CDLL でロードする
#
# extern "C" の役割:
#   C++ コンパイラは関数名をマングリング (装飾) する。
#   例: tracking_step → _ZN8trackdlo13tracking_stepE...
#   ctypes は関数名の文字列でシンボルを探すため、マングリングされると見つけられない。
#   extern "C" を付けると C の ABI でシンボルが公開され、名前がそのまま残る。
#
# Windows クロスコンパイル方法 (Ubuntu から):
#   1. ツールチェーンのインストール
#      sudo apt install mingw-w64
#
#   2. pure_trackdlo を Windows 向けにビルド
#      mkdir -p build_win/pure_trackdlo && cd build_win/pure_trackdlo
#      cmake ../../src/pure_trackdlo \
#        -DCMAKE_TOOLCHAIN_FILE=../../src/trackdlo_cdll/cmake/toolchain-mingw64.cmake \
#        -DCMAKE_INSTALL_PREFIX=../../install_win
#      make && make install
#
#   3. trackdlo_cdll を Windows 向けにビルド
#      mkdir -p build_win/trackdlo_cdll && cd build_win/trackdlo_cdll
#      cmake ../../src/trackdlo_cdll \
#        -DCMAKE_TOOLCHAIN_FILE=../../src/trackdlo_cdll/cmake/toolchain-mingw64.cmake \
#        -DCMAKE_PREFIX_PATH=../../install_win
#      make
#      → trackdlo_c.dll が生成される
# =============================================================================

# =============================================================================
# ビルドターゲット切り替え
#
#   OFF (デフォルト) : Linux 向け共有ライブラリ  → libtrackdlo_c.so
#   ON               : Windows 向け共有ライブラリ → trackdlo_c.dll
#
# ON にする場合は mingw64 ツールチェーンと組み合わせる。
# cmake コマンドに -DCMAKE_TOOLCHAIN_FILE=cmake/toolchain-mingw64.cmake を追加するか、
# このファイルを直接編集して ON にしてからビルドする。
#
# [マクロの役割]
#   BUILD_FOR_WINDOWS が ON のとき、コンパイラに -DTRACKDLO_WINDOWS を渡す。
#   trackdlo_c_api.h はこのマクロを検知して TRACKDLO_API を
#   __declspec(dllexport) に展開する。
#
#   Linux では共有ライブラリのシンボルはデフォルトで公開されるが、
#   Windows の DLL ではシンボルに __declspec(dllexport) を付けないと
#   ctypes が関数を見つけられない。TRACKDLO_API はこの差を吸収するマクロ。
# =============================================================================
option(BUILD_FOR_WINDOWS "Build for Windows cross-compilation (mingw64)" OFF)

find_package(pure_trackdlo REQUIRED)
find_package(Eigen3 3.3 REQUIRED NO_MODULE)

# SHARED: 共有ライブラリとしてビルドする。
# Linux では libtrackdlo_c.so、Windows では trackdlo_c.dll が生成される。
# CMake が CMAKE_SYSTEM_NAME に応じて拡張子とプレフィックスを自動で決める。
add_library(trackdlo_c
    SHARED
    src/trackdlo_c_api.cpp
)

if(BUILD_FOR_WINDOWS)
    # -DTRACKDLO_WINDOWS をコンパイラに渡す。
    # trackdlo_c_api.h の TRACKDLO_API マクロが __declspec(dllexport) に展開される。
    target_compile_definitions(trackdlo_c PRIVATE TRACKDLO_WINDOWS)
endif()

target_include_directories(trackdlo_c PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# pure_trackdlo は静的ライブラリ (.a) だが POSITION_INDEPENDENT_CODE ON 済みのため
# 共有ライブラリへの静的リンクが可能。
target_link_libraries(trackdlo_c
    PRIVATE
    pure_trackdlo
    Eigen3::Eigen
)

install(
    TARGETS trackdlo_c
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)

install(
    DIRECTORY include/
    DESTINATION include
)

