llvm_map_components_to_libnames(llvm_libs support core irreader)

cuda_add_executable(driver_test driver_test.cu)
target_link_libraries( driver_test ${OpenCV_LIBS} )

add_executable(tutorial tutorial.cc)
target_link_libraries( tutorial ${llvm_libs} )

add_library(KernelInfoPass MODULE kernel_info_pass.cc)
set_target_properties(KernelInfoPass PROPERTIES
    COMPILE_FLAGS "-fno-rtti"
)
target_link_libraries( KernelInfoPass glog )
