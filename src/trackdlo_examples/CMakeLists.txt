cmake_minimum_required(VERSION 3.14)
project(trackdlo_examples)

set(CMAKE_CXX_STANDARD 17)
add_compile_options(-O3)

# pure_trackdlo はライブラリとしてビルド済みであること。
# colcon build --packages-up-to trackdlo_examples で依存が自動解決される。
find_package(pure_trackdlo REQUIRED)
find_package(Eigen3 3.3 REQUIRED NO_MODULE)

# ---- C++ サンプル ----
# クロスコンパイル時 (Windows 向けビルド) はホストで実行できないため除外する
if(NOT CMAKE_CROSSCOMPILING)
    add_executable(ex_hello_trackdlo     cpp/hello_trackdlo.cpp)
    add_executable(ex_state_lifecycle    cpp/state_lifecycle.cpp)

    target_link_libraries(ex_hello_trackdlo  pure_trackdlo)
    target_link_libraries(ex_state_lifecycle pure_trackdlo)
endif()
