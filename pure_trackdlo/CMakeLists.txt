cmake_minimum_required(VERSION 3.14)
project(pure_trackdlo)

set(CMAKE_CXX_STANDARD 17)
add_compile_options(-O3)

find_package(Eigen3 3.3 REQUIRED NO_MODULE)

add_library(pure_trackdlo
    src/trackdlo.cpp
    src/utils.cpp
)
target_include_directories(pure_trackdlo PUBLIC include)
target_link_libraries(pure_trackdlo Eigen3::Eigen)

enable_testing()
find_package(GTest REQUIRED)

add_executable(test_utils tests/test_utils.cpp)
target_link_libraries(test_utils pure_trackdlo GTest::gtest_main)

add_executable(test_trackdlo tests/test_trackdlo.cpp)
target_link_libraries(test_trackdlo pure_trackdlo GTest::gtest_main)

include(GoogleTest)
gtest_discover_tests(test_utils)
gtest_discover_tests(test_trackdlo)
