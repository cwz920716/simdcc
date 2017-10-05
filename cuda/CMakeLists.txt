cuda_add_executable(driver_test driver_test.cu)
target_link_libraries( driver_test ${OpenCV_LIBS} )

cuda_add_executable(axpy axpy.cu)

cuda_add_executable(gaussian_elimination gaussian_elimination.cu)
target_link_libraries( gaussian_elimination glog )
