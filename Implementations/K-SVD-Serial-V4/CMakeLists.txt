cmake_minimum_required(VERSION 3.10)
project(ksvd_serial_v4)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find Eigen3
find_package(Eigen3 REQUIRED)

# Find OpenCV (core modules only)
find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs)

# Create executable
add_executable(main main.cpp omp.cpp omp.hpp image_process.cpp image_process.hpp)

# Link Eigen3
target_link_libraries(main Eigen3::Eigen)

# Link OpenCV
target_link_libraries(main ${OpenCV_LIBS})

# Include directories
target_include_directories(main PRIVATE ${EIGEN3_INCLUDE_DIR})
target_include_directories(main PRIVATE ${OpenCV_INCLUDE_DIRS})
