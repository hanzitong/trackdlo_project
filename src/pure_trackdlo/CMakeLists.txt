cmake_minimum_required(VERSION 3.14)
project(pure_trackdlo)

set(CMAKE_CXX_STANDARD 17)
add_compile_options(-O3)

find_package(Eigen3 3.3 REQUIRED NO_MODULE)

# ---- ライブラリ本体 ----
# Linux (.a) でも Windows クロスコンパイル (.a → .dll に静的リンク) でも
# 常にビルドする。クロスコンパイル判定は CMAKE_CROSSCOMPILING で行う。
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
# クロスコンパイル時 (Windows 向けビルド) はホスト上で実行できないため除外する
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
