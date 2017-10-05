cuda_add_executable(axpy axpy.cu)

cuda_add_executable(gaussian_elimination gaussian_elimination.cu)
target_link_libraries( gaussian_elimination glog )
