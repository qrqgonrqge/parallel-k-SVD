cmake_minimum_required(VERSION 3.16)
project(aksvd_parallel LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs)
find_package(OpenMP REQUIRED)

add_executable(main
    main.cpp
    omp.cpp
    ksvd.cpp
    distorter.cpp
    image_process.cpp
    synth.cpp
)

target_link_libraries(main PRIVATE
    ${OpenCV_LIBS}
    OpenMP::OpenMP_CXX
)

target_include_directories(main PRIVATE
    ${OpenCV_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR}
)

target_compile_options(main PRIVATE -O3 -DNDEBUG)
