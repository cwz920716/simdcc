#include <cuda.h>
#include <builtin_types.h>
#include <iostream>
#include <map>
#include <cstdlib>
#include <stack>
#include <string>
#include <sys/stat.h>
#include <vector>

#include "glog/logging.h"

#include "drvapi_error_string.h"
#include "cuda_dynamic_runtime.h"

namespace gpuvm {

// This will output the proper CUDA error strings in the event that a CUDA host call returns an error
#define checkCudaErrors(err)  __checkCudaErrors (err, __FILE__, __LINE__)

// These are the inline versions for all of the SDK helper functions
void __checkCudaErrors( CUresult err, const char *file, const int line )
{
    if( CUDA_SUCCESS != err) {
        fprintf(stderr, "checkCudaErrors() Driver API error = %04d \"%s\" from file <%s>, line %i.\n",
                err, getCudaDrvErrorString(err), file, line );
        exit(-1);
    }
}

CUdevice cudaDeviceInit()
{
    CUdevice cuDevice = 0;
    int deviceCount = 0;
    CUresult err = cuInit(0);
    char name[100];
    int major=0, minor=0;

    if (CUDA_SUCCESS == err)
        checkCudaErrors(cuDeviceGetCount(&deviceCount));
    if (deviceCount == 0) {
        fprintf(stderr, "cudaDeviceInit error: no devices supporting CUDA\n");
        exit(-1);
    }
    checkCudaErrors(cuDeviceGet(&cuDevice, 0));
    cuDeviceGetName(name, 100, cuDevice);
    printf("Using CUDA Device [0]: %s\n", name);

    checkCudaErrors( cuDeviceComputeCapability(&major, &minor, cuDevice) );
    if (major < 2) {
        fprintf(stderr, "Device 0 is not sm_20 or later\n");
        exit(-1);
    }
    return cuDevice;
}


CUresult initCUDA(CUdevice *phDevice,
                  CUcontext *phContext)
{
    // Initialize
    *phDevice = cudaDeviceInit();

    // Create context on the device
    checkCudaErrors(cuCtxCreate(phContext, 0, *phDevice));

    return CUDA_SUCCESS;
}

char *loadProgramSource(const char *filename, size_t *size) 
{
    struct stat statbuf;
    FILE *fh;
    char *source = NULL;
    *size = 0;
    fh = fopen(filename, "rb");
    if (fh) {
        stat(filename, &statbuf);
        source = (char *) malloc(statbuf.st_size+1);
        if (source) {
            fread(source, statbuf.st_size, 1, fh);
            source[statbuf.st_size] = 0;
            *size = statbuf.st_size+1;
        }
    }
    else {
        fprintf(stderr, "Error reading file %s\n", filename);
        exit(-1);
    }
    return source;
}

enum FileType {
  LL,
  PTX
};

class Program {
 public:
  Program(const std::string &filename, FileType type):
    filename_(filename), type_(type), source_(nullptr), size_(0),
    cuda_module_(0) {}

  bool Load() {
    source_ = loadProgramSource(filename_.c_str(), &size_);

    // Load the PTX
    auto ptx = GetPTXString();
    if (ptx == nullptr) {
      return false;
    }

    checkCudaErrors(cuModuleLoadDataEx(&cuda_module_, ptx, 0, 0, 0));
    return true;
  }

  void Unload() {
    if (cuda_module_) {
      checkCudaErrors(cuModuleUnload(cuda_module_));
      cuda_module_ = 0;
    }

    if (source_) {
      free(source_);
      source_ = nullptr;
    }
  }

  char *GetPTXString() {
    if (type_ == PTX) {
      return source_;
    }

    return nullptr;
  }

  CUresult GetFunctionByName(CUfunction *function, const char *name) {
    return cuModuleGetFunction(function, cuda_module_, name);
  }

 private:
  std::string filename_;
  FileType type_;
  char *source_;
  size_t size_;
  bool loaded_;
  CUmodule cuda_module_;
};

struct KernelLaunchParams {
 public:
  KernelLaunchParams(dim3 gridDim, dim3 blockDim,
                     size_t sharedMem, cudaStream_t stream):
    gridDim_(gridDim), blockDim_(blockDim), sharedMem_(sharedMem),
    stream_(stream) {}

  dim3 gridDim_;
  dim3 blockDim_;
  size_t sharedMem_;
  cudaStream_t stream_;
  std::vector<void *> arguments_;
};

// TODO(wcui): make it thread-safe!
class GPUMachine {
 public:
  GPUMachine() {}

  void AddProgram(const std::string &filename, FileType type = PTX) {
    auto prog = new Program(filename, type);
    programs_.push_back(prog);
  }

  void Setup() {
    initCUDA(&device_, &context_);
    for (auto p : programs_) {
      p->Load();
    }
  }

  void Destroy() {
    for (auto p : programs_) {
      p->Unload();
    }
    checkCudaErrors(cuCtxDestroy(context_));
  }

  void ConfigureKernelLaunch(dim3 gridDim, dim3 blockDim,
                             size_t sharedMem, cudaStream_t stream) {
    auto params = new KernelLaunchParams(gridDim, blockDim, sharedMem, stream);
    execution_stack_.push(params);
  }

  void SetupArgument(void *arg) {
    KernelLaunchParams *top = execution_stack_.top();
    top->arguments_.push_back(arg);
  }

  cudaError_t GetFunction(CUfunction *function, const char *name) {
    for (auto p : programs_) {
      if (p->GetFunctionByName(function, name) == CUDA_SUCCESS) {
        return cudaSuccess;
      }
    }
    return cudaErrorInvalidDeviceFunction;
  }

  CUfunction GetRegisteredFunction(void *host_fun) {
    CHECK(kernel_HtoD_.find(host_fun) != kernel_HtoD_.end());
    return kernel_HtoD_[host_fun];
  }

  cudaError_t LaunchKernel(CUfunction kernel) {
    KernelLaunchParams *params = execution_stack_.top();
    auto gridSizeX = params->gridDim_.x;
    auto gridSizeY = params->gridDim_.y;
    auto gridSizeZ = params->gridDim_.z;
    auto blockSizeX = params->blockDim_.x;
    auto blockSizeY = params->blockDim_.y;
    auto blockSizeZ = params->blockDim_.z;
    unsigned int sharedMem = params->sharedMem_;
    cudaStream_t stream = params->stream_;
    void **args = params->arguments_.data();
    checkCudaErrors(cuLaunchKernel(kernel, gridSizeX, gridSizeY, gridSizeZ,
                                   blockSizeX, blockSizeY, blockSizeZ,
                                   sharedMem, stream, args, NULL));
    return cudaSuccess;
  }

  void RegisterFunction(void *host_fun, const char *name) {
    CUfunction device_fun;
    auto func_valid = GetFunction(&device_fun, name);
    if (func_valid != cudaSuccess) {
      LOG(FATAL) << "Cannot register a non-exist device function!";
      return;
    }

    kernel_HtoD_[host_fun] = device_fun;
  }

 private:
  CUdevice device_;
  CUcontext context_;
  std::map<void *, CUfunction> kernel_HtoD_;
  std::vector<Program *> programs_;
  std::stack<KernelLaunchParams *> execution_stack_;
};

}  // namespace gpuvm

static gpuvm::GPUMachine processGPU;

void PrepareDynamicCuda(void) {
  processGPU.AddProgram("kernel.ptx");
  processGPU.Setup();
}

void DestroyDynamicCuda(void) {
  processGPU.Destroy();
}

cudaError_t
dynamicCudaConfigureCall(dim3 gridDim, dim3 blockDim,
                         size_t sharedMem, cudaStream_t stream) {
  processGPU.ConfigureKernelLaunch(gridDim, blockDim, sharedMem, stream);
  return cudaSuccess;
}

cudaError_t
dynamicCudaSetupArgument(const void* arg, size_t size, size_t offset) {
  // TODO(wcui): memory leak
  char *arg_copy = new char[size];
  memcpy(arg_copy, arg, size);
  processGPU.SetupArgument(arg_copy);
  return cudaSuccess;
}

cudaError_t dynamicCudaLaunchByName(const char* name) {
  CUfunction function;
  auto func_valid = processGPU.GetFunction(&function, name);
  if (func_valid != cudaSuccess) {
    return func_valid;
  }

  return processGPU.LaunchKernel(function);
}

void dynamicCudaRegisterFunction(void *hostFun, const char *name) {
  processGPU.RegisterFunction(hostFun, name);
}

cudaError_t dynamicCudaLaunch(void *hostFun) {
  CUfunction function = processGPU.GetRegisteredFunction(hostFun);
  return processGPU.LaunchKernel(function);
}
