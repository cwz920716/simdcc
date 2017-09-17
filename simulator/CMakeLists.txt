add_library(memory SHARED memory.cc)
add_executable(main_test main_test.cc)
target_link_libraries(main_test memory glog)
