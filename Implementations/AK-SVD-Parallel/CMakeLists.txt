cmake_minimum_required(VERSION 3.16)
project(aksvd_parallel LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs)
find_package(TBB REQUIRED)
find_package(Eigen3 QUIET)
if(NOT Eigen3_FOUND)
    set(EIGEN3_INCLUDE_DIR "/usr/include/eigen3" CACHE PATH "Eigen3 include dir")
endif()

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
    TBB::tbb
)

target_include_directories(main PRIVATE
    ${OpenCV_INCLUDE_DIRS}
    $<IF:$<TARGET_EXISTS:Eigen3::Eigen>,,${EIGEN3_INCLUDE_DIR}>
)
if(TARGET Eigen3::Eigen)
    target_link_libraries(main PRIVATE Eigen3::Eigen)
endif()

target_compile_options(main PRIVATE -O3 -DNDEBUG)
target_link_options(main PRIVATE -fstack-protector)
