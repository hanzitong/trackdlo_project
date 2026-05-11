cmake_minimum_required(VERSION 3.14)
project(evaluation)

set(CMAKE_CXX_STANDARD 17)
add_compile_options(-O3)

find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(OpenCV REQUIRED)
find_package(PCL 1.8 REQUIRED COMPONENTS common)

add_library(evaluation
    src/evaluator.cpp
)
target_include_directories(evaluation PUBLIC
    include
    ${CMAKE_CURRENT_SOURCE_DIR}/../pure_trackdlo/include
)
target_link_libraries(evaluation
    Eigen3::Eigen
    ${OpenCV_LIBS}
    ${PCL_LIBRARIES}
)

# pure_trackdlo も一緒にビルドする (utils.h/cpp に依存するため)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../pure_trackdlo
                 ${CMAKE_CURRENT_BINARY_DIR}/pure_trackdlo)
target_link_libraries(evaluation pure_trackdlo)
