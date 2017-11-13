cmake_minimum_required(VERSION 3.4)

project(SimdCC)

set (CMAKE_CXX_STANDARD 11)

set(Boost_USE_STATIC_LIBS OFF)
set(Boost_USE_MULTITHREADED ON)
set(Boost_USE_STATIC_RUNTIME OFF)

find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Boost REQUIRED)

find_package(LLVM REQUIRED CONFIG)
add_definitions(${LLVM_DEFINITIONS})
include_directories(${LLVM_INCLUDE_DIRS})
link_directories(${LLVM_LIBRARY_DIRS})

include_directories( ${OpenCV_INCLUDE_DIRS} )

include_directories( ${Boost_INCLUDE_DIR} )

include_directories(common)
include_directories(cub-1.7.4)

if (CUDA_FOUND)
  message("CUDA found")
  set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode arch=compute_35,code=sm_35;-std=c++11;--expt-extended-lambda)
  add_subdirectory(cuda)
else()
  message("Error: CUDA not found!")
endif()

add_subdirectory(compiler)
#add_subdirectory(emulator)
#add_subdirectory(synthesis)
