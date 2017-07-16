cmake_minimum_required(VERSION 2.8)

project(SimdSim)

<<<<<<< HEAD
find_package(CUDA REQUIRED)
find_package( OpenCV REQUIRED )

include_directories( ${OpenCV_INCLUDE_DIRS} )
=======
find_package(CUDA)
>>>>>>> 9245312cc3e5c713b5bd2bf23eacdeec5bfe67b0

if (CUDA_FOUND)
  message("CUDA found")
  set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode arch=compute_20,code=sm_20)
  add_subdirectory(cuda)
else()
	message("Error: CUDA not found!")
endif()


set(CMAKE_CXX_FLAG "-g -Wall")
