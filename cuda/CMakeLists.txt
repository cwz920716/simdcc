cuda_add_executable(driver_test driver_test.cu)
target_link_libraries( driver_test ${OpenCV_LIBS} )

cuda_add_executable(axpy axpy.cu)

cuda_add_executable(gaussian_elimination gaussian_elimination.cu)
target_link_libraries( gaussian_elimination glog )

cuda_add_executable(bfs bfs.cu)

cuda_add_executable(test_glang test_glang.cu)
target_link_libraries( test_glang glog )
