cmake_minimum_required(VERSION 3.10)
project(2d_ksvd_baseline)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find Eigen3
find_package(Eigen3 REQUIRED)

# Create executable
add_executable(2d_baseline 2d_baseline.cpp)

# Link Eigen3
target_link_libraries(2d_baseline Eigen3::Eigen)

# Include directories
target_include_directories(2d_baseline PRIVATE ${EIGEN3_INCLUDE_DIR})
