cmake_minimum_required(VERSION 3.14)
project(pure_trackdlo)

set(CMAKE_CXX_STANDARD 17)
add_compile_options(-O3)

# =============================================================================
# [事前知識] ファイル形式とツールチェーン
#
# ツールチェーンとは、コンパイラ・リンカ・標準ライブラリなど
# ビルドに必要なツール一式のこと。ターゲット OS に応じて使い分ける。
#
# コンパイラはソースコードを機械語命令に変換してファイルに格納するが、
# このファイルの包装形式 (機械語やシンボルテーブルをどう並べるか) は
# ターゲット OS によって決まる。
#
#   ELF (Executable and Linkable Format): Linux が使うファイル形式 (.so, .a)
#   PE/COFF (Portable Executable):        Windows が使うファイル形式 (.dll, .lib)
#
# なお、機械語命令そのものは CPU アーキテクチャで決まるため
# x86_64 なら Linux / Windows で同じ内容になる。
# ELF / PE はあくまで「その機械語をどういう構造のファイルに入れるか」という形式である。
#
#   ELF ファイルの構造:
#     [ ELF ヘッダ          ]  ← 識別子・アーキテクチャ情報
#     [ .text セクション    ]  ← 機械語命令 (x86_64 なら PE と同内容)
#     [ .data セクション    ]  ← 初期化済みグローバル変数
#     [ シンボルテーブル    ]  ← 関数名とアドレスの対応表
#
#   PE ファイルの構造:
#     [ DOS ヘッダ + PE ヘッダ ]  ← Windows 独自の識別子・構造
#     [ .text セクション       ]  ← 機械語命令 (x86_64 なら ELF と同内容)
#     [ .data セクション       ]
#     [ エクスポートテーブル   ]  ← DLL が公開する関数の一覧
#
# OS のローダーはヘッダを読んでどこに何を配置するかを判断するため、
# ヘッダ形式が違うと読めない。同じ機械語が入っていても使えない。
#
# =============================================================================
# [ビルドの切り替え方]
#
# このCMakeLists.txt自体は Linux 向けでも Windows 向けでも共通して使用可能。
# ターゲットを切り替えるには cmake コマンドに渡すツールチェーンを変えるだけでよい。
# ソースコード (trackdlo.cpp / utils.cpp) も CMakeLists.txt も変更不要。
#
#   Linux 向けビルド (デフォルト):
#     cmake ..
#     make
#     → ELF 形式の libpure_trackdlo.a が生成される
#
#   Windows 向けクロスビルド (Ubuntu 上で mingw64 を使う):
#     cmake .. -DCMAKE_TOOLCHAIN_FILE=../trackdlo_cdll/cmake/toolchain-mingw64.cmake
#     make
#     → PE/COFF 形式の libpure_trackdlo.a が生成される
#
# ELF と PE/COFF は互換性がないため、trackdlo_c.dll を Windows 向けにビルドする場合は
# リンク先の pure_trackdlo もこの手順で Windows 向けにビルドし直すこと。
# =============================================================================

find_package(Eigen3 3.3 REQUIRED NO_MODULE)

# ---- ライブラリ本体 ----
# Linux / Windows どちらのツールチェーンでも同じ記述でビルドできる。
add_library(pure_trackdlo
    src/trackdlo.cpp
    src/utils.cpp
)
# $<BUILD_INTERFACE:...> はビルド時のパス、$<INSTALL_INTERFACE:...> はインストール後のパス
# downstream が find_package(pure_trackdlo) したときに正しい include パスが渡る
target_include_directories(pure_trackdlo PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(pure_trackdlo Eigen3::Eigen)
# trackdlo_cdll が SHARED ライブラリとして pure_trackdlo を静的リンクするために -fPIC が必要
set_target_properties(pure_trackdlo
    PROPERTIES POSITION_INDEPENDENT_CODE ON
)

# ---- テスト ----
# サンプルプログラムは trackdlo_examples パッケージに移動済み。
# クロスコンパイル時 (Windows 向けビルド) はホスト上で実行できないため除外する。
# CMAKE_CROSSCOMPILING は CMake 組み込み変数。ツールチェーンファイルを指定して
# ホストと異なるターゲット向けにビルドすると CMake が自動で TRUE にセットする。
if(NOT CMAKE_CROSSCOMPILING)
    enable_testing()
    find_package(GTest REQUIRED)

    add_executable(test_utils tests/test_utils.cpp)
    target_link_libraries(test_utils pure_trackdlo GTest::gtest_main)

    add_executable(test_trackdlo tests/test_trackdlo.cpp)
    target_link_libraries(test_trackdlo pure_trackdlo GTest::gtest_main)

    include(GoogleTest)
    gtest_discover_tests(test_utils)
    gtest_discover_tests(test_trackdlo)
endif()

# ---- インストール & cmake package config 生成 ----
# colcon build 後、downstream が find_package(pure_trackdlo REQUIRED) できるようにする
install(TARGETS pure_trackdlo
    EXPORT pure_trackdlo-targets
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)
install(DIRECTORY include/ DESTINATION include)

include(CMakePackageConfigHelpers)
configure_package_config_file(
    cmake/pure_trackdlo-config.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/pure_trackdlo-config.cmake
    INSTALL_DESTINATION lib/cmake/pure_trackdlo
)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/pure_trackdlo-config.cmake
    DESTINATION lib/cmake/pure_trackdlo
)
install(EXPORT pure_trackdlo-targets
    FILE pure_trackdlo-targets.cmake
    DESTINATION lib/cmake/pure_trackdlo
)
