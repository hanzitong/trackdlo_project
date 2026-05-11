cmake_minimum_required(VERSION 3.14)
project(preprocessing)

set(CMAKE_CXX_STANDARD 17)
add_compile_options(-O3)

find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(OpenCV REQUIRED)
find_package(PCL 1.8 REQUIRED COMPONENTS common filters)

add_library(preprocessing
    src/preprocessing.cpp
)
target_include_directories(preprocessing PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(preprocessing
    Eigen3::Eigen
    ${OpenCV_LIBS}
    ${PCL_LIBRARIES}
)

enable_testing()
find_package(GTest REQUIRED)

add_executable(test_preprocessing tests/test_preprocessing.cpp)
target_link_libraries(test_preprocessing preprocessing GTest::gtest_main)

include(GoogleTest)
gtest_discover_tests(test_preprocessing)

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
