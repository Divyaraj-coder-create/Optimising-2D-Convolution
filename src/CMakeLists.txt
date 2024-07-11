cmake_minimum_required(VERSION 3.10)
project(your_project_name)

# Find OpenMP package
find_package(OpenMP REQUIRED)

add_library(studentlib main.cpp)

target_include_directories(studentlib PUBLIC ${CMAKE_SOURCE_DIR}/include)

# Link OpenMP library
target_link_libraries(studentlib PRIVATE OpenMP::OpenMP_CXX)

target_compile_options(studentlib PRIVATE -Ofast -march=native -mfma -fopenmp )
