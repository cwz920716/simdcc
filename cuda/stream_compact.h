#pragma once

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <thrust/scan.h>

#include <cub/cub.cuh>

int NPOT = 0;
int SIZE = 1 << 16 - NPOT;
tcc::Blob<int> a(SIZE), b(SIZE), c(SIZE);

template<typename T>
int cmpArrays(int n, T *a, T *b) {
    for (int i = 0; i < n; i++) {
        if (a[i] != b[i]) {
            printf("    a[%d] = %d, b[%d] = %d\n", i, a[i], i, b[i]);
            return 1;
        }
    }
    return 0;
}

void genArray(int n, int *a, int maxval) {
    srand(0);

    for (int i = 0; i < n; i++) {
        a[i] = rand() % maxval;
        // printf("    a[%d] = %d", i, a[i]);
    }
    printf("\n");
}

TEST(StreamCompact, Prepare) {
  int *Pa_h = a.mutable_cpu_data();
  int *Pa_d = a.mutable_gpu_data();
  genArray(SIZE-1, Pa_h, 50);
  Pa_h[SIZE-1] = 0;
  tcc::caffe_gpu_memcpy(SIZE, Pa_h, Pa_d);

  int *Pb_h = b.mutable_cpu_data();
  int *Pb_d = b.mutable_gpu_data();
  tcc::caffe_set(SIZE-1, 0, Pb_h);
  tcc::caffe_gpu_set(SIZE-1, 0, Pb_d);
  int *Pc_h = c.mutable_cpu_data();
  int *Pc_d = c.mutable_gpu_data();
  tcc::caffe_set(SIZE-1, 0, Pc_h);
  tcc::caffe_gpu_set(SIZE-1, 0, Pc_d);
}

/**
 * CPU scan (prefix sum).
 */
void scan(int n, int *odata, const int *idata) {
    // TODO
  int psum = 0;
  for (int i = 0; i < n; i++) {
    odata[i] = psum;
    psum += idata[i];
  }
}

/**
 * CPU stream compaction without using the scan function.
 *
 * @returns the number of elements remaining after compaction.
 */
int compactWithoutScan(int n, int *odata, const int *idata) {
  // TODO
  int ip = 0, op = 0;
  for (; ip < n; ip++) {
    if (idata[ip] != 0) {
      odata[op] = idata[ip];
      op++;
    }
  }
  return op;
}

/**
 * CPU stream compaction using scan and scatter, like the parallel version.
 *
 * @returns the number of elements remaining after compaction.
 */
int compactWithScan(int n, int *odata, const int *idata) {
  // TODO
  int *mark = (int *)malloc(n * sizeof(int));
  int *idx = (int *)malloc((n+1) * sizeof(int));
  for (int i = 0; i < n; i++) {
    mark[i] = (idata[i] != 0) ? 1 : 0;
  }
  scan(n, idx, mark);
  idx[n] = idx[n-1] + mark[n-1];
  for (int i = 0; i < n; i++) {
    if (idx[i] < idx[i+1]) {
      odata[idx[i]] = idata[i];
    }
  }

  return idx[n];
}

TEST(StreamCompact, CPUs_scan) {
  int *Pa = a.mutable_cpu_data();
  genArray(SIZE-1, Pa, 256);
  const int *Pa_h = a.cpu_data();
  int *Pb_h = b.mutable_cpu_data();
  int *Pc_h = c.mutable_cpu_data();
  compactWithoutScan(SIZE, Pb_h, Pa_h);
  compactWithScan(SIZE, Pc_h, Pa_h);
  EXPECT_EQ(cmpArrays(SIZE, Pb_h, Pc_h), 0);
}

inline int ilog2(int x)
{
    int lg = 0;
    while (x >>= 1)
    {
        ++lg;
    }
    return lg;
}

inline int ilog2ceil(int x)
{
    return ilog2(x - 1) + 1;
}

const int blockSize = 128;

namespace Naive {

// TODO: __global__
__global__ void kernInclusiveToExclusiveScan(int n, int * odata, int * idata) {
  int index = threadIdx.x + (blockIdx.x * blockDim.x);
  if (index >= n) {
    return;
  }
  odata[index] = (index == 0 ) ? 0 : idata[index - 1];
}

__global__ void kernNaiveScan(int n, int depth, int *odata, int *idata) {
  int index = threadIdx.x + (blockIdx.x * blockDim.x);
  if (index >= n) {
    return;
  }
  int p2d = 1 << (depth-1);
  odata[index] = (index < p2d) ? idata[index] :
                                 (idata[index] + idata[index- p2d]);
}

/**
 * Performs prefix-sum (aka scan) on idata, storing the result into odata.
 */
void scan(int n, int *odata, const int *idata) {
  // TODO
  int n2 = 1 << ilog2ceil(n);
  dim3 fullBlocksPerGrid((n2 + blockSize - 1) / blockSize);
  int * dev_data;
  int * dev_data2;
  cudaMalloc((void**)&dev_data, n2 * sizeof(int));
  cudaMalloc((void**)&dev_data2, n2 * sizeof(int));
  cudaMemcpy((void*)dev_data, (void*)idata, n * sizeof(int), cudaMemcpyHostToDevice);

  for (int i = 1; i <= ilog2ceil(n); i++) {
    kernNaiveScan <<<fullBlocksPerGrid, blockSize>>> (n, i, dev_data2, dev_data);
    int * tempPtr = dev_data;
    dev_data = dev_data2;
    dev_data2 = tempPtr;
  }
  kernInclusiveToExclusiveScan <<<fullBlocksPerGrid, blockSize>>> (n, dev_data2, dev_data);

  cudaMemcpy((void*)odata, (void*)dev_data2, n * sizeof(int), cudaMemcpyDeviceToHost);
  cudaFree(dev_data);
  cudaFree(dev_data2);
}

}

TEST(StreamCompact, GPUNaiveScan) {
  int *Pa = a.mutable_cpu_data();
  genArray(SIZE-1, Pa, 256);
  const int *Pa_h = a.cpu_data();
  int *Pb_h = b.mutable_cpu_data();
  int *Pc_h = c.mutable_cpu_data();
  scan(SIZE, Pc_h, Pa_h);
  Naive::scan(SIZE, Pb_h, Pa_h);
  EXPECT_EQ(cmpArrays(SIZE, Pb_h, Pc_h), 0);
}

namespace Common {

/**
 * Maps an array to an array of 0s and 1s for stream compaction. Elements
 * which map to 0 will be removed, and elements which map to 1 will be kept.
 */
__global__ void kernMapToBoolean(int n, int *bools, const int *idata) {
	int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if (index >= n) {
		return;
	}
	bools[index] = (idata[index] == 0) ? 0 : 1;
}

/**
 * Performs scatter on an array. That is, for each element in idata,
 * if bools[idx] == 1, it copies idata[idx] to odata[indices[idx]].
 */
__global__ void kernScatter(int n, int *odata,
        const int *idata, const int *bools, const int *indices) {
	int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if (index >= n) {
		return;
	}
	if (bools[index] != 0) {
		odata[indices[index]] = idata[index];
	}
}

}  // namespace Common

namespace Efficient {

__global__ void kernScanUpsweep(int n, int d, int * data) {
  int index = threadIdx.x + (blockIdx.x * blockDim.x);
  int d1 = d + 1;
  if (index >= (n >> d1)) {
    return;
  }
  int group = 1 << d1;
  int k = index * group;
  data[k + group - 1] += data[k + (group >> 1) - 1];
}

__global__ void kernScanDownsweep(int n, int d, int *data) {
  int index = threadIdx.x + (blockIdx.x * blockDim.x);
  int d1 = d + 1;
  int group = 1 << d1;
  if (index >= (n >> d1)) {
    return;
  }
  int sub = 1 << d;
  int k = index * group;
  int t = data[k + sub - 1];
  data[k+sub-1] = data[k+group-1];
  data[k+group-1] += t;
}

__global__ void _kernScanDownsweep(int n, int d, int * data) {
	int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if (index >= (n >> d)) {
		return;
	}
	int k = index << d;
	int t = data[k + (1 << d) - 1];
	data[k + (1 << d) - 1] += data[k + (1 << (d - 1)) - 1];
	data[k + (1 << (d - 1)) - 1] = t;
}

// TODO: __global__

/**
 * Performs prefix-sum (aka scan) on idata, storing the result into odata.
 */
void scan(int n, int *odata, const int *idata) {
  // TODO
  int * dev_data;
  int logCeil = ilog2ceil(n);
  int nCeil = 1 << logCeil;

  cudaMalloc((void**)&dev_data, nCeil * sizeof(int));
  cudaMemset((void*)dev_data, 0, nCeil * sizeof(int));
  cudaMemcpy((void*)dev_data, (void*)idata, n * sizeof(int), cudaMemcpyHostToDevice);

  for (int i = 0; i < logCeil; i++) {
    int gridSize = ((nCeil >> (i+1)) + blockSize - 1) / blockSize;
    kernScanUpsweep <<< gridSize, blockSize >>> (nCeil, i, dev_data);
  }
  cudaMemset(dev_data + nCeil - 1, 0, sizeof(int));
  for (int i = logCeil-1; i >= 0; i--) {
    int gridSize = ((nCeil >> (i+1)) + blockSize - 1) / blockSize;
    kernScanDownsweep << <gridSize, blockSize >> >(nCeil, i, dev_data);
  }

  cudaMemcpy((void*)odata, (void*)dev_data, n * sizeof(int), cudaMemcpyDeviceToHost);
  cudaFree(dev_data);
}

/**
 * Performs stream compaction on idata, storing the result into odata.
 * All zeroes are discarded.
 *
 * @param n      The number of elements in idata.
 * @param odata  The array into which to store elements.
 * @param idata  The array of elements to compact.
 * @returns      The number of elements remaining after compaction.
 */
int compact(int n, int *odata, const int *idata) {
  // TODO
  	int * dev_bools;
	int * dev_idata;
	int * dev_odata;
	int * dev_indices;
	cudaMalloc((void**)&dev_bools, n * sizeof(int));
	cudaMalloc((void**)&dev_indices, n * sizeof(int));
	cudaMalloc((void**)&dev_idata, n * sizeof(int));
	cudaMalloc((void**)&dev_odata, n * sizeof(int));

	// Map to booleans
	cudaMemcpy((void*)dev_idata, (void*)idata, n * sizeof(int), cudaMemcpyHostToDevice);
	Common::kernMapToBoolean<<< (n + blockSize - 1) / blockSize, blockSize >>>(n, dev_bools, dev_idata);
	int * temp = (int *)malloc(n * sizeof(int));
	cudaMemcpy((void*)temp, (void*)dev_bools, n * sizeof(int), cudaMemcpyDeviceToHost);

	// Do exclusive scan
	scan(n, temp, temp);
	int compactedCount = temp[n - 1] + ((idata[n - 1] == 0) ? 0 : 1);

	// Scatter
	cudaMemcpy((void*)dev_indices, (void*)temp, n * sizeof(int), cudaMemcpyHostToDevice);
	Common::kernScatter <<< (n + blockSize - 1) / blockSize, blockSize >>>(n, dev_odata, dev_idata, dev_bools, dev_indices);
	cudaMemcpy((void*)odata, (void*)dev_odata, n * sizeof(int), cudaMemcpyDeviceToHost);

	free(temp);
	cudaFree(dev_bools);
	cudaFree(dev_idata);
	cudaFree(dev_odata);
	cudaFree(dev_indices);

	return compactedCount;
}

}  // namespace Efficient

TEST(StreamCompact, GPUBetterScan) {
  int *Pa = a.mutable_cpu_data();
  genArray(SIZE-1, Pa, 256);
  const int *Pa_h = a.cpu_data();
  int *Pb_h = b.mutable_cpu_data();
  int *Pc_h = c.mutable_cpu_data();
  scan(SIZE, Pc_h, Pa_h);
  Efficient::scan(SIZE, Pb_h, Pa_h);
  EXPECT_EQ(cmpArrays(SIZE, Pb_h, Pc_h), 0);
}

TEST(StreamCompact, GPUCompact) {
  int *Pa = a.mutable_cpu_data();
  genArray(SIZE-1, Pa, 256);
  const int *Pa_h = a.cpu_data();
  int *Pb_h = b.mutable_cpu_data();
  int *Pc_h = c.mutable_cpu_data();
  int bret = Efficient::compact(SIZE, Pb_h, Pa_h);
  int cret = compactWithScan(SIZE, Pc_h, Pa_h);
  EXPECT_EQ(bret, cret);
  EXPECT_EQ(cmpArrays(bret, Pb_h, Pc_h), 0);
}

void printCudaArray(int n, int *a) {
  int *Ha = (int *) malloc(n * sizeof(int));
  cudaMemcpy((void *)Ha, (void *)a, n * sizeof(int), cudaMemcpyDeviceToHost);
  for (int i = 0; i < n; i++) {
    std::cout << Ha[i] << "  ";
  }
  std::cout << std::endl;
  free(Ha);
}

void printArray(int n, int *a) {
  for (int i = 0; i < n; i++) {
    std::cout << a[i] << "  ";
  }
  std::cout << std::endl;
}


namespace Thrust {

/**
 * Performs prefix-sum (aka scan) on idata, storing the result into odata.
 */
void scan(int n, int *odata, const int *idata) {
  // TODO use `thrust::exclusive_scan`
  // example: for device_vectors dv_in and dv_out:
  // thrust::exclusive_scan(dv_in.begin(), dv_in.end(), dv_out.begin());
  thrust::device_vector<int> dev_thrust_idata(idata, idata + n);
  thrust::device_vector<int> dev_thrust_odata(n);
  thrust::exclusive_scan(dev_thrust_idata.begin(), dev_thrust_idata.end(), dev_thrust_odata.begin());
  thrust::copy(dev_thrust_odata.begin(), dev_thrust_odata.end(), odata);
}

// assuming all positives

__global__ void radix2bools(int n, int bit, int *odata0, int *odata1, int *idata) {
  int tid = threadIdx.x + (blockIdx.x * blockDim.x);
  if (tid >= n) {
    return;
  }
  int d = idata[tid];
  int mask = (1 << bit);
  int is_one = ((d & mask) != 0) ? 1 : 0;
  int is_zero = 1 - is_one;
  odata0[tid] = is_zero;
  odata1[tid] = is_one; 
}

__global__ void scatter(int n, int *odata, const int *idata, const int *bools, const int *indices) {
  int tid = threadIdx.x + (blockIdx.x * blockDim.x);
  if (tid >= n) {
    return;
  }
  if (bools[tid] != 0) {
    odata[indices[tid]] = idata[tid];
  }
}

__global__ void scatter1(int n, int *odata, int *idata, const int *bools, const int *indices, const int *user) {
  int tid = threadIdx.x + (blockIdx.x * blockDim.x);
  if (tid >= n) {
    return;
  }
  int offset = user[0];
  if (bools[tid] != 0) {
    odata[indices[tid] + offset] = idata[tid];
  }
}

void radixSort(int n, int *odata, const int *idata) {
  thrust::device_vector<int> idata_d(idata, idata+n);
  thrust::device_vector<int> zeros(n); // dv_zeros[i] = 1 if Nth bit of dv_in[i] = 0
  thrust::device_vector<int> ones(n); // dv_ones[i] = 1 if Nth bit of dv_in[i] = 1
  thrust::device_vector<int> odata_d(n), scans(n);
  int *sort_in, *sort_out;
  sort_in = raw_pointer_cast(&idata_d[0]);
  sort_out = raw_pointer_cast(&odata_d[0]);
  int *pzeros = raw_pointer_cast(&zeros[0]);
  int *pones = raw_pointer_cast(&ones[0]);
  int *pscans = raw_pointer_cast(&scans[0]);
  int gridSize = (n + blockSize - 1) / blockSize;
  std::cout << std::endl;
  for (int i = 0; i < 31; i++) {
    // printCudaArray(n, sort_in);
    radix2bools <<<gridSize, blockSize>>> (n, i, pzeros, pones, sort_in);
    // printCudaArray(n, pzeros);
    // printCudaArray(n, pones);
    thrust::exclusive_scan(zeros.begin(), zeros.end(), scans.begin());
    // printCudaArray(n, pscans);
    thrust::device_vector<int> user(1);
    user[0] = zeros[n-1] + scans[n-1];
    int *puser = raw_pointer_cast(&user[0]);
    scatter <<<gridSize, blockSize>>> (n, sort_out, sort_in, pzeros, pscans);
    // printCudaArray(n, sort_out);
    thrust::exclusive_scan(ones.begin(), ones.end(), scans.begin());
    scatter1 <<<gridSize, blockSize>>> (n, sort_out, sort_in, pones, pscans, puser);
    // printCudaArray(n, sort_out);
    int * temp = sort_in; sort_in = sort_out; sort_out = temp;
  }
  cudaMemcpy((void *)odata, (void *)sort_in, n * sizeof(int), cudaMemcpyDeviceToHost);
}

}  // namespace Thrust

TEST(StreamCompact, RadixSort) {
  int *Pa = a.mutable_cpu_data();
  genArray(SIZE-1, Pa, 256);
  const int *Pa_h = a.cpu_data();
  int *Pb_h = b.mutable_cpu_data();
  Thrust::radixSort(SIZE, Pb_h, Pa_h);
  for (int i = 1; i < SIZE; i++) {
    EXPECT_GE(Pb_h[i], Pb_h[i-1]);
  }
}

__global__ void histogramKernel(int num_bins, int *bins, int n, const int *idata) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= n) {
    return;
  }

  extern __shared__ int histo_s[];
  CUDA_BLOCK_LOOP(i, num_bins) {
    histo_s[i] = 0;
  }
  __syncthreads();

  CUDA_KERNEL_LOOP(i, n) {
    int binIdx = idata[i] / 4;
    atomicAdd(&histo_s[binIdx], 1);
  }
  __syncthreads();

  CUDA_BLOCK_LOOP(i, num_bins) {
    atomicAdd(&bins[i], histo_s[i]);
  }
}

void histogram(int *bins, int n, const int *idata) {
  for (int i = 0; i < 8; i++) {
    bins[i] = 0;
  }
  for (int i = 0; i < n; i++) {
    bins[idata[i]/4]++;
  }
}

using namespace thrust;

void histogram_gpu(int *bins, int n, const int *idata) {
  thrust::device_vector<int> idata_d(idata, idata+n);
  thrust::device_vector<int> bins_d(8, 0);
  int *Pidata = thrust::raw_pointer_cast(&idata_d[0]);
  int *Pbins = thrust::raw_pointer_cast(&bins_d[0]);
  int gridSize = (n + blockSize - 1) / blockSize;
  // printCudaArray(8, Pbins);
  histogramKernel <<< gridSize, blockSize, 8 * sizeof(int)  >>> (8, Pbins, n, Pidata);
  // printCudaArray(8, Pbins);
  cudaMemcpy((void *)bins, (void *)Pbins, 8 * sizeof(int), cudaMemcpyDeviceToHost);
}

TEST(StreamCompact, Histogram) {
  int *Pa = a.mutable_cpu_data();
  genArray(SIZE, Pa, 16);
  const int *Pa_h = a.cpu_data();
  int *Pb_h = b.mutable_cpu_data();
  int *Pc_h = c.mutable_cpu_data();
  histogram(Pb_h, SIZE, Pa_h);
  histogram_gpu(Pc_h, SIZE, Pa_h);
  EXPECT_EQ(cmpArrays(8, Pb_h, Pc_h), 0);
}
