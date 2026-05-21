cmake_minimum_required(VERSION 3.14)
project(trackdlo_cdll)

set(CMAKE_CXX_STANDARD 17)
add_compile_options(-O3)

find_package(pure_trackdlo REQUIRED)
find_package(Eigen3 3.3 REQUIRED NO_MODULE)

# ================================================================
# [ビルドの全体像]
#
# 生成物: libtrackdlo_c.so
#
#   trackdlo_c_api.cpp (C ラッパー関数)
#       + libpure_trackdlo.a (アルゴリズム本体、静的リンクで .so に吸収)
#   ↓ コンパイル & リンク
#   libtrackdlo_c.so  ← Python が ctypes.CDLL でロードするファイル
#
# pybind11 との違い:
#   pybind11 は Python 専用の拡張モジュール (.cpython-310-...so)。
#   こちらは純粋な共有ライブラリで、C, Python, Julia 等どこからでも呼べる。
#
# extern "C" の役割:
#   C++ のコンパイラは関数名をマングリング (装飾) する。
#   例: tracking_step → _ZN8trackdlo13tracking_stepE...
#   これだと ctypes から名前で関数を探せない。
#   extern "C" を付けると C の ABI (名前そのまま) でシンボルが公開される。
# ================================================================

# SHARED: 共有ライブラリ (.so) として作る
# pure_trackdlo は静的ライブラリ (.a) だが、POSITION_INDEPENDENT_CODE ON
# が設定済みのため .so への静的リンクが可能。
add_library(trackdlo_c SHARED src/trackdlo_c_api.cpp)

target_include_directories(trackdlo_c PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(trackdlo_c PRIVATE pure_trackdlo Eigen3::Eigen)

install(TARGETS trackdlo_c LIBRARY DESTINATION lib)
install(DIRECTORY include/ DESTINATION include)
