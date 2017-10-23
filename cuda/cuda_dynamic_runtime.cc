#include <math.h>
#include <cuda.h>
#include <builtin_types.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>

#include "drvapi_error_string.h"

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
                  CUcontext *phContext,
                  const char *ptx)
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

 private:
  std::string filename_;
  FileType type_;
  char *source_;
  size_t size_;
  bool loaded_;
  CUmodule cuda_module_;
};

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
    checkCudaErrors(cuCtxDestroy(hContext));
  }

 private:
  CUdevice device_;
  CUcontext context_;
  std::vector<Program *> programs_;
};

static GPUMachine processGPU;

void PrepareDynamicCuda(void) {
  processGPU.AddProgram("kernel.ptx");
  processGPU.Setup();
}

void DestroyDynamicCuda(void) {
  processGPU.Destroy();
}

}  // namespace gpuvm
