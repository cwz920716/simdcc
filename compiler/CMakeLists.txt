llvm_map_components_to_libnames(llvm_libs asmparser bitreader bitwriter core ipo irreader linker object support transformutils)

add_executable(tutorial tutorial.cc)
target_link_libraries( tutorial ${llvm_libs} )

add_executable(llvm-culink llvm-culink.cc)
set_target_properties(llvm-culink PROPERTIES
    COMPILE_FLAGS "-fno-rtti"
)
target_link_libraries( llvm-culink ${llvm_libs} ${Boost_LIBRARIES} )

set(cuda_gpuvm_src
  kernel_info_pass.cc
  cond_branch_analysis_pass.cc
)

add_library(gpuvm MODULE ${cuda_gpuvm_src})
set_target_properties(gpuvm PROPERTIES
    COMPILE_FLAGS "-fno-rtti"
)
target_link_libraries( gpuvm glog )
