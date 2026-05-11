cmake_minimum_required(VERSION 3.14)
project(evaluation)

set(CMAKE_CXX_STANDARD 17)
add_compile_options(-O3)

find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(OpenCV REQUIRED)
find_package(PCL 1.8 REQUIRED COMPONENTS common)
# colcon build 後の install space に pure_trackdlo が配置されるので find_package で取得する
find_package(pure_trackdlo REQUIRED)

add_library(evaluation
    src/evaluator.cpp
)
target_include_directories(evaluation PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(evaluation
    Eigen3::Eigen
    ${OpenCV_LIBS}
    ${PCL_LIBRARIES}
    pure_trackdlo  # cmake target として export されている
)

# ---- インストール & cmake package config 生成 ----
install(TARGETS evaluation
    EXPORT evaluation-targets
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)
install(DIRECTORY include/ DESTINATION include)

include(CMakePackageConfigHelpers)
configure_package_config_file(
    cmake/evaluation-config.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/evaluation-config.cmake
    INSTALL_DESTINATION lib/cmake/evaluation
)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/evaluation-config.cmake
    DESTINATION lib/cmake/evaluation
)
install(EXPORT evaluation-targets
    FILE evaluation-targets.cmake
    DESTINATION lib/cmake/evaluation
)
