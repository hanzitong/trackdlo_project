cmake_minimum_required(VERSION 3.14)
project(preprocessing_cdll)

set(CMAKE_CXX_STANDARD 17)
add_compile_options(-O3)

# =============================================================================
# [ビルドの全体像]
#
# 生成物: libpreprocessing_c.so (Linux) / preprocessing_c.dll (Windows)
#
#   preprocessing_c_api.cpp (extern "C" で公開した C ラッパー関数)
#       + libpreprocessing.a (前処理アルゴリズム本体、静的リンクで吸収)
#   ↓ コンパイル & リンク
#   libpreprocessing_c.so / preprocessing_c.dll  ← Python が ctypes.CDLL でロードする
#
# =============================================================================
# ビルドモード切り替え
#
#   DEFAULT (両フラグ OFF):
#     Ubuntu 通常ビルド → libpreprocessing_c.so (PCL + OpenCV を動的リンク)
#
#   BUILD_STANDALONE=ON:
#     持ち運び可能な Linux .so → libpreprocessing_c.so
#     - PCL を除去 (カスタム VoxelGrid で代替)
#     - OpenCV を静的リンク (.a) → PCL/OpenCV の .so が先方 PC になくても動作する
#     - 前提: preprocessing を BUILD_STANDALONE=ON でビルド & インストール済みであること
#
#   BUILD_FOR_WINDOWS=ON:
#     Ubuntu から Windows 向けクロスコンパイル → preprocessing_c.dll
#     - PCL を除去
#     - PREPROC_WINDOWS 定義 → __declspec(dllexport) 有効化
#
# ─── 通常 Ubuntu ビルド ───────────────────────────────────────────────────────
#   colcon build
#
# ─── スタンドアロン .so ビルド (持ち運び用) ─────────────────────────────────
#   mkdir -p build_standalone/preprocessing && cd build_standalone/preprocessing
#   cmake ../../src/preprocessing \
#     -DBUILD_STANDALONE=ON \
#     -DCMAKE_INSTALL_PREFIX=../../install_standalone
#   make && make install && cd ../..
#
#   mkdir -p build_standalone/preprocessing_cdll && cd build_standalone/preprocessing_cdll
#   cmake ../../src/preprocessing_cdll \
#     -DBUILD_STANDALONE=ON \
#     -DCMAKE_PREFIX_PATH=../../install_standalone
#   make && cd ../..
#   → build_standalone/preprocessing_cdll/libpreprocessing_c.so
#
# ─── Windows クロスコンパイル ────────────────────────────────────────────────
#   docs/build_guide.md 参照
# =============================================================================

option(BUILD_FOR_WINDOWS "Build for Windows cross-compilation (mingw64)" OFF)
option(BUILD_STANDALONE  "Build self-contained .so: no PCL, static OpenCV" OFF)

find_package(preprocessing REQUIRED)
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(OpenCV REQUIRED)

if(BUILD_STANDALONE)
    # スタンドアロンビルド: OpenCV を静的 .a でリンクする
    # find_package(OpenCV) はインクルードパス取得のみに使い、
    # ライブラリは .a ファイルを直接指定する。
    # Ubuntu の libopencv-dev には -fPIC 付きの .a が含まれるため .so に静的リンクできる。
    find_library(OPENCV_CORE_STATIC    NAMES libopencv_core.a    REQUIRED)
    find_library(OPENCV_IMGPROC_STATIC NAMES libopencv_imgproc.a REQUIRED)
    set(_OPENCV_LINK_LIBS ${OPENCV_CORE_STATIC} ${OPENCV_IMGPROC_STATIC})
    message(STATUS "Standalone build: static OpenCV")
    message(STATUS "  opencv_core:    ${OPENCV_CORE_STATIC}")
    message(STATUS "  opencv_imgproc: ${OPENCV_IMGPROC_STATIC}")

    # preprocessing の cmake ターゲットは INTERFACE_LINK_LIBRARIES に
    # opencv_core;opencv_imgproc;... 等の .so ターゲット群を持つ (推移依存)。
    # このままにすると静的 .a と動的 .so が二重リンクされてしまうため、
    # INTERFACE_LINK_LIBRARIES を Eigen3 のみに上書きして .so の推移を遮断する。
    set_target_properties(preprocessing PROPERTIES
        INTERFACE_LINK_LIBRARIES "Eigen3::Eigen"
    )
elseif(BUILD_FOR_WINDOWS)
    # Windows クロスコンパイル: find_package が返す .a (mingw向け) をそのまま使う
    set(_OPENCV_LINK_LIBS ${OpenCV_LIBS})
else()
    # 通常 Ubuntu ビルド: PCL + 動的 OpenCV
    find_package(PCL 1.8 REQUIRED COMPONENTS common filters)
    set(_OPENCV_LINK_LIBS ${OpenCV_LIBS})
endif()

add_library(preprocessing_c
    SHARED
    src/preprocessing_c_api.cpp
)

if(BUILD_FOR_WINDOWS)
    # Windows DLL のシンボルエクスポートに必要
    target_compile_definitions(preprocessing_c PRIVATE PREPROC_WINDOWS)
endif()

target_include_directories(preprocessing_c PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

if(BUILD_STANDALONE)
    # INTERFACE_LINK_LIBRARIES を上書きしたことで OpenCV の cmake ターゲット経由の
    # インクルードパス伝播が切れるため、OpenCV_INCLUDE_DIRS を直接追加する。
    target_include_directories(preprocessing_c PRIVATE ${OpenCV_INCLUDE_DIRS})
endif()

# preprocessing は -fPIC 付きでビルド済みのため共有ライブラリへの静的リンクが可能
target_link_libraries(preprocessing_c
    PRIVATE
    preprocessing
    Eigen3::Eigen
    ${_OPENCV_LINK_LIBS}
)

if(NOT BUILD_STANDALONE AND NOT BUILD_FOR_WINDOWS)
    target_link_libraries(preprocessing_c PRIVATE ${PCL_LIBRARIES})
endif()

install(
    TARGETS preprocessing_c
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)

install(
    DIRECTORY include/
    DESTINATION include
)
