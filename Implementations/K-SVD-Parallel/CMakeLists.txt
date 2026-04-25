cmake_minimum_required(VERSION 3.10)
project(ksvd_parallel)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs)
find_package(OpenMP REQUIRED)

add_executable(main
    main.cpp
    omp.cpp        omp.hpp
    ksvd.cpp       ksvd.hpp
    distorter.cpp  distorter.hpp
    image_process.cpp image_process.hpp
)

target_link_libraries(main Eigen3::Eigen)
target_link_libraries(main ${OpenCV_LIBS})
target_link_libraries(main OpenMP::OpenMP_CXX)

target_include_directories(main PRIVATE ${EIGEN3_INCLUDE_DIR})
target_include_directories(main PRIVATE ${OpenCV_INCLUDE_DIRS})

target_compile_options(main PRIVATE -O3 -march=native -DNDEBUG)
