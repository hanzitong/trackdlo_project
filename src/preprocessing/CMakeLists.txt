cmake_minimum_required(VERSION 3.14)
project(preprocessing)

set(CMAKE_CXX_STANDARD 17)
add_compile_options(-O3)

# =============================================================================
# ビルドターゲット切り替え
#
#   OFF (デフォルト) : Ubuntu/Linux 向け → libpreprocessing.a (PCL 使用)
#   ON               : Windows 向け     → libpreprocessing.a (PCL 不使用)
#
# ON にすると PREPROCESSING_WINDOWS が定義され、preprocessing.cpp 内の
# #ifdef PREPROCESSING_WINDOWS ブロックが有効になる。
# PCL は mingw64 向けの apt パッケージが存在しないため、
# Windows ビルドでは std::unordered_map を使ったカスタム VoxelGrid に差し替える。
#
# ─── Ubuntu ビルド (デフォルト) ───────────────────────────────────────────────
#   colcon build  または
#   mkdir -p build && cd build && cmake .. && make
#
# ─── Windows クロスコンパイル (Ubuntu から) ──────────────────────────────────
#
#   依存ライブラリの入手:
#     - Eigen3    : ヘッダオンリーなので Ubuntu のものをそのまま使える
#     - PCL       : BUILD_FOR_WINDOWS=ON のとき自動スキップ (不要)
#     - OpenCV    : Windows 向けバイナリが別途必要。以下いずれかで取得する。
#
#       方法A) 公式 OpenCV Windows リリースの MinGW ビルドを使う
#         OpenCV 公式サイトからWindows 用リリースをダウンロードし、
#         opencv/build/x64/mingw/lib/ にある *.a と *.dll.a を
#         install_win/lib/ に、opencv/build/include/ を install_win/include/ にコピーする。
#         (同梱の OpenCVConfig.cmake を install_win/lib/cmake/opencv4/ に置く)
#
#       方法B) vcpkg で mingw-static トリプレットを使う
#         vcpkg install opencv4[core,imgproc]:x64-mingw-static
#         CMAKE_PREFIX_PATH に vcpkg のインストールパスを指定する
#
#   1. mingw-w64 ツールチェーンのインストール
#      sudo apt install mingw-w64
#
#   2. preprocessing を Windows 向けにビルド
#      mkdir -p build_win/preprocessing && cd build_win/preprocessing
#      cmake ../../src/preprocessing \
#        -DBUILD_FOR_WINDOWS=ON \
#        -DCMAKE_TOOLCHAIN_FILE=../../src/trackdlo_cdll/cmake/toolchain-mingw64.cmake \
#        -DCMAKE_PREFIX_PATH=../../install_win \
#        -DCMAKE_INSTALL_PREFIX=../../install_win
#      make && make install
# =============================================================================
# option() は set() と違い -DBUILD_FOR_WINDOWS=ON のようにコマンドラインから上書きできる
option(BUILD_FOR_WINDOWS "Build without PCL for Windows cross-compilation" OFF)

find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(OpenCV REQUIRED)

if(NOT BUILD_FOR_WINDOWS)
    find_package(PCL 1.8 REQUIRED COMPONENTS common filters)
endif()

add_library(preprocessing
    src/preprocessing.cpp
)

if(BUILD_FOR_WINDOWS)
    target_compile_definitions(preprocessing PUBLIC PREPROCESSING_WINDOWS)
endif()

target_include_directories(preprocessing PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(preprocessing
    Eigen3::Eigen
    ${OpenCV_LIBS}
)

if(NOT BUILD_FOR_WINDOWS)
    target_link_libraries(preprocessing ${PCL_LIBRARIES})
endif()

# preprocessing_cdll が SHARED ライブラリとして静的リンクするために -fPIC が必要
set_target_properties(preprocessing
    PROPERTIES POSITION_INDEPENDENT_CODE ON
)

# テストは Ubuntu 環境のみ (mingw64 クロスコンパイル時はスキップ)
if(NOT BUILD_FOR_WINDOWS)
    enable_testing()
    find_package(GTest REQUIRED)

    add_executable(test_preprocessing tests/test_preprocessing.cpp)
    target_link_libraries(test_preprocessing preprocessing GTest::gtest_main)

    include(GoogleTest)
    gtest_discover_tests(test_preprocessing)
endif()

# ---- インストール & cmake package config 生成 ----
install(TARGETS preprocessing
    EXPORT preprocessing-targets
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)
install(DIRECTORY include/ DESTINATION include)

include(CMakePackageConfigHelpers)
configure_package_config_file(
    cmake/preprocessing-config.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/preprocessing-config.cmake
    INSTALL_DESTINATION lib/cmake/preprocessing
)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/preprocessing-config.cmake
    DESTINATION lib/cmake/preprocessing
)
install(EXPORT preprocessing-targets
    FILE preprocessing-targets.cmake
    DESTINATION lib/cmake/preprocessing
)
