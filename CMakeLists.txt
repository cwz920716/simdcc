cmake_minimum_required(VERSION 2.8)

project(SimdSim)

find_package(CUDA REQUIRED)
find_package( OpenCV REQUIRED )

include_directories( ${OpenCV_INCLUDE_DIRS} )

if (CUDA_FOUND)
  message("CUDA found")
  set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode arch=compute_20,code=sm_20)
  add_subdirectory(cuda)
else()
	message("Error: CUDA not found!")
endif()


set(CMAKE_CXX_FLAG "-g -Wall")
