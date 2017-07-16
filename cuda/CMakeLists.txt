cuda_add_executable(driver_test driver_test.cu)
target_link_libraries( driver_test ${OpenCV_LIBS} )
