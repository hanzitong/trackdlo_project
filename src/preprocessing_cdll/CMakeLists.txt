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
# trackdlo_cdll と同じ構造。違いは以下:
#   - pure_trackdlo の代わりに preprocessing をラップする
#   - cv::Mat を受け取る関数は uint8_t* / uint16_t* で橋渡しする
#   - 可変長出力 (点群) は呼び元が最大サイズのバッファを渡し、実際の要素数を int* で受け取る
# =============================================================================

# BUILD_FOR_WINDOWS: trackdlo_cdll と同じ切り替え方式
# ON にすると PREPROC_WINDOWS が定義され、preprocessing_c_api.h の PREPROC_API が
# __declspec(dllexport) に展開される。
set(BUILD_FOR_WINDOWS OFF)

find_package(preprocessing REQUIRED)
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(OpenCV REQUIRED)
find_package(PCL 1.8 REQUIRED COMPONENTS common filters)

add_library(preprocessing_c
    SHARED
    src/preprocessing_c_api.cpp
)

if(BUILD_FOR_WINDOWS)
    target_compile_definitions(preprocessing_c PRIVATE PREPROC_WINDOWS)
endif()

target_include_directories(preprocessing_c PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# preprocessing は -fPIC 付きでビルド済みのため共有ライブラリへの静的リンクが可能
# OpenCV / PCL は preprocessing が推移的に依存しているが、
# 共有ライブラリのリンク時に明示的に指定しないと undefined symbol になる場合があるため追加する
target_link_libraries(preprocessing_c
    PRIVATE
    preprocessing
    Eigen3::Eigen
    ${OpenCV_LIBS}
    ${PCL_LIBRARIES}
)

install(
    TARGETS preprocessing_c
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)

install(
    DIRECTORY include/
    DESTINATION include
)
