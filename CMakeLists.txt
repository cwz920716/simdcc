cmake_minimum_required(VERSION 3.4)

project(SimdSim)

set (CMAKE_CXX_STANDARD 11)

find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Boost REQUIRED)

find_package(LLVM REQUIRED CONFIG)
add_definitions(${LLVM_DEFINITIONS})
include_directories(${LLVM_INCLUDE_DIRS})
link_directories(${LLVM_LIBRARY_DIRS})

include_directories( ${OpenCV_INCLUDE_DIRS} )

add_subdirectory(compiler)
add_subdirectory(emulator)

if (CUDA_FOUND)
  message("CUDA found")
  set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode arch=compute_35,code=sm_35)
  add_subdirectory(cuda)
else()
  message("Error: CUDA not found!")
endif()
