llvm_map_components_to_libnames(llvm_libs support core irreader)

cuda_add_executable(driver_test driver_test.cu)
target_link_libraries( driver_test ${OpenCV_LIBS} )

add_executable(tutorial tutorial.cc)
target_link_libraries( tutorial ${llvm_libs} )

set(cuda_gpuvm_src
  kernel_info_pass.cc
  cond_branch_analysis_pass.cc
)

add_library(gpuvm MODULE ${cuda_gpuvm_src})
set_target_properties(gpuvm PROPERTIES
    COMPILE_FLAGS "-fno-rtti"
)
target_link_libraries( gpuvm glog )
