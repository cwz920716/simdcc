#pragma once

#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <queue>

struct CSRGraph {
  int nodes_size;
  int edges_size;
  int *edges; // nodes_size + 1
  int *dest;
};

using namespace std;
using Edges = unordered_set<int>;
using MemGraph = unordered_map< int, Edges >;  // directed graph

void initMemGraph(MemGraph &g, int n, int e) {
  srand(0);
  for (int i = 0; i < e; i++) {
    int x = rand() % n;
    int y = rand() % n;
    if (g[x].count(y) == 0) {
      g[x].insert(y);
    } else {
      i--;
    }
  }
}

void convert(int n, int e, MemGraph &g1, CSRGraph &g2) {
  g2.nodes_size = n;
  int edge_sum = 0;
  for (int i = 0; i < n; i++) {
    g2.edges[i] = edge_sum;
    for (int x : g1[i]) {
      g2.dest[edge_sum] = x;
      edge_sum++;
    }
  }
  CHECK_EQ(e, edge_sum);
  g2.edges[n] = edge_sum;
  g2.nodes_size = n;
  g2.edges_size = edge_sum;
}

void printGraph(CSRGraph &g) {
  for (int i = 0; i < g.nodes_size; i++) {
    for (int j = g.edges[i]; j < g.edges[i+1]; j++) {
      cout << "(" << i << "->" << g.dest[j] << ")" << endl;
    }
  }
}

void printGraph(int n, MemGraph &g) {
  for (int i = 0; i < n; i++) {
    for (int j : g[i]) {
      cout << "(" << i << "->" << j << ")" << endl;
    }
  }
}

void bfs_cpu(CSRGraph &g, int src, int *label) {
  std::unordered_set<int> mark;
  std::queue<int> WL;
  WL.push(src);
  mark.insert(src);
  label[src] = 0;
  while (!WL.empty()) {
    int hd = WL.front();
    WL.pop();
    int e_start = g.edges[hd];
    int e_size = g.edges[hd+1] - g.edges[hd];
    for (int i = 0; i < e_size; i++) {
      int neig = g.dest[i+e_start];
      if (mark.count(neig) == 0) {
        WL.push(neig);
        mark.insert(neig);
        label[neig] = label[hd] + 1;
      }
    }
  }
}

TEST(BFS, Cpu) {
  int n = 10;
  int e = 15;
  MemGraph Gx;
  initMemGraph(Gx, n, e);
  CSRGraph G;
  G.edges = new int[n+1];
  G.dest = new int[e];
  int *label = new int[n];
  memset(label, 0, n *sizeof(int));
  convert(n, e, Gx, G);
  // printGraph(G);
  bfs_cpu(G, 0, label);
  // printArray(n, label);
  delete[] G.edges;
  delete[] G.dest;
  delete[] label;
}

#define BLOCK_QUEUE_SIZE (1024)

__global__ void bfs_gpu(int n, int *WL, int *n_out, int *out_WL, int *edges, int *dest, int *label, int *visited) {
  __shared__ int block_WL[BLOCK_QUEUE_SIZE];
  __shared__ int n_out_s;
  __shared__ int block_n_out;

  if (threadIdx.x == 0) n_out_s = 0;
  __syncthreads();

  CUDA_KERNEL_LOOP(idx, n) {
    int v = WL[idx];
    for (int i = edges[v]; i < edges[v+1]; i++) {
      int u = dest[i];
      // printf("%d->%d\n", v, u);
      int visited_before = atomicExch(&visited[u], 1);
      if (!visited_before) {
        label[u] = label[v] + 1;
        int next_slot = atomicAdd(&n_out_s, 1);
        if (next_slot >= BLOCK_QUEUE_SIZE) {
          // Block Queue is full
          // printf("overflow\n");
          next_slot = atomicAdd(n_out, 1);
          out_WL[next_slot] = u;
          n_out_s = BLOCK_QUEUE_SIZE;
        } else {
          block_WL[next_slot] = u;
        }
      }
    }
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    block_n_out = atomicAdd(n_out, n_out_s);
    // printf("%d %d\n", block_n_out, n_out_s);
  }
  __syncthreads();

  CUDA_BLOCK_LOOP(idx, n_out_s) {
    out_WL[block_n_out + idx] = block_WL[idx];
  }
  return;
}

#define PTR(dv) ( thrust::raw_pointer_cast(&dv[0]) )
#define PTR_AT(dv, i) ( thrust::raw_pointer_cast(&dv[i]) )

TEST(BFS, Gpu) {
  int n = 150000;
  int e = 750000;
  MemGraph Gx;
  initMemGraph(Gx, n, e);
  CSRGraph G;
  G.edges = new int[n+1];
  G.dest = new int[e];
  int *label = new int[n];
  memset(label, 0, n *sizeof(int));
  int *visited = new int[n];
  memset(visited, 0, n *sizeof(int));

  convert(n, e, Gx, G);
  // printGraph(G);
  int src = 0;
  visited[src] = 1;

  device_vector<int> edges_d(G.edges, G.edges+n+1);
  device_vector<int> dest_d(G.dest, G.dest+e);
  device_vector<int> label_d(label, label+n);
  device_vector<int> visited_d(visited, visited+n);
  device_vector<int> WL_in(e, 0);
  device_vector<int> WL_out(e, 0);
  device_vector<int> n_out(1, 0);
  int n_wl = 1;
  WL_in[0] = src;
  int *wl_in = PTR(WL_in);
  int *wl_out = PTR(WL_out);
cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);
cudaEventRecord(start);
  while (n_wl > 0) {
    int gridSize = (n_wl + blockSize - 1) / blockSize;
    bfs_gpu <<<gridSize, blockSize>>> (n_wl, wl_in, PTR(n_out), wl_out, PTR(edges_d), PTR(dest_d), PTR(label_d), PTR(visited_d));
    // std::cout << "after loop: \n";
    n_wl = n_out[0];
    EXPECT_LE(n_wl, WL_out.size());
    n_out[0] = 0;
    int *tmp = wl_in; wl_in = wl_out; wl_out = tmp;
    // std::cout << "WL=" << n_wl << std::endl;
    // printCudaArray(n, PTR(label_d));
    // break;
  }
cudaEventRecord(stop);
cudaEventSynchronize(stop);
float milliseconds = 0;
cudaEventElapsedTime(&milliseconds, start, stop);
std::cout << "Time: " << milliseconds << " (ms)." << endl;

  bfs_cpu(G, src, label);
  // printArray(n, label);

  for (int i = 0; i < n; i++) {
    // std::cout << i << endl;
    CHECK_EQ(label[i], label_d[i]);
  }
  delete[] G.edges;
  delete[] G.dest;
  delete[] label;
  delete[] visited;
}

/**TODO:
 *   1. Warp level queue.
 *   2. aggragate atomics
 *   3. coarse-grained, warp-based
 *   4. coarse-grained, CTA-based
 *   5. fine-grained, scan-based
 *   6. scan + warp + CTA
 **/
#define WARP_SZ 32
__device__
inline int lane_id(void) { return threadIdx.x % WARP_SZ; }
__device__
inline int warp_id(void) { return threadIdx.x / WARP_SZ; }
__device__ int warp_bcast(int v, int leader) { return __shfl(v, leader); }

__global__ void bfs_gpu_o0(int n, int *WL, int *n_out, int *out_WL, int *edges, int *dest, int *label, int *visited) {
  CUDA_KERNEL_LOOP(idx, n) {
    int v = WL[idx];
    for (int i = edges[v]; i < edges[v+1]; i++) {
      int u = dest[i];
      // printf("%d->%d\n", v, u);
      int visited_before = atomicExch(&visited[u], 1);
      if (!visited_before) {
        label[u] = label[v] + 1;
        int next_slot = atomicAdd(n_out, 1);
        out_WL[next_slot] = u;
      }
    }
  }
}

#define WARP_QUEUE_SIZE (BLOCK_QUEUE_SIZE / 8)

__global__ void bfs_gpu_o1(int n, int *WL, int *n_out, int *out_WL, int *edges, int *dest, int *label, int *visited) {
  __shared__ int warp_WL[8][WARP_QUEUE_SIZE];
  __shared__ int n_out_s;
  __shared__ int n_out_w[8];
  __shared__ int n_out_w_psum[8];
  __shared__ int block_n_out;

  if (threadIdx.x == 0) {
    n_out_s = 0;
    for (int i = 0; i < 8; i++)
      n_out_w[i] = 0;
  }
  __syncthreads();

  int wid = lane_id() % 8;  // warp id
  CUDA_KERNEL_LOOP(idx, n) {
    int v = WL[idx];
    for (int i = edges[v]; i < edges[v+1]; i++) {
      int u = dest[i];
      // printf("[%d]: %d->%d\n", wid, v, u);
      int visited_before = atomicExch(&visited[u], 1);
      if (!visited_before) {
        label[u] = label[v] + 1;
        int next_slot = atomicAdd(&n_out_w[wid], 1);
        if (next_slot >= WARP_QUEUE_SIZE) {
          // warp Queue is full, direct store to global queue
          next_slot = atomicAdd(n_out, 1);
          out_WL[next_slot] = u;
          n_out_w[wid] = WARP_QUEUE_SIZE;
        } else {
          warp_WL[wid][next_slot] = u;
        }
      }
    }
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    // load warp level queue to block level queue
    n_out_s = 0;
    for (int i = 0; i < 8; i++) {
      n_out_w_psum[i] = n_out_s;
      n_out_s += n_out_w[i];
    }
    block_n_out = atomicAdd(n_out, n_out_s);
  }
  __syncthreads();

  CUDA_BLOCK_LOOP(idx, 8 * WARP_QUEUE_SIZE) {
    int qid = idx / WARP_QUEUE_SIZE;
    int qpos = idx % WARP_QUEUE_SIZE;
    if (qpos < n_out_w[qid]) {
      out_WL[block_n_out + qpos + n_out_w_psum[qid]] = warp_WL[qid][qpos];
    }
  }
  return;
}

__device__
int warp_agg_reserve(int *pool, int cond=1) {
  int mask = __ballot(cond);
  int size = __popc(mask);
  int leader = __ffs(mask) - 1;
  int res;
  if (lane_id() == leader) {
    res = atomicAdd(pool, size);
  }
  res = warp_bcast(res, leader);
  int offset = __popc(mask & ((1 << lane_id()) - 1));
  return res + offset;
}

__global__ void bfs_gpu_o2(int n, int *WL, int *n_out, int *out_WL, int *edges, int *dest, int *label, int *visited) {
  CUDA_KERNEL_LOOP(idx, n) {
    int v = WL[idx];
    for (int i = edges[v]; i < edges[v+1]; i++) {
      int u = dest[i];
      // printf("%d->%d\n", v, u);
      int visited_before = atomicExch(&visited[u], 1);
      int next_slot = warp_agg_reserve(n_out, !visited_before);
      if (!visited_before) {
        label[u] = label[v] + 1;
        out_WL[next_slot] = u;
      }
    }
  }
}

#define BLOCK_SZ (64)

__global__ void bfs_gpu_o3(int n, int *WL, int *n_out, int *out_WL, int *edges, int *dest, int *label, int *visited) {
  __shared__ int block_WL[BLOCK_QUEUE_SIZE];
  __shared__ int n_out_s;
  __shared__ int block_n_out;
  int tid = threadIdx.x + blockIdx.x * blockDim.x;

  if (threadIdx.x == 0) n_out_s = 0;
  __syncthreads();

  for (int w = 0; w < WARP_SZ; w++) {
    int leader = blockIdx.x * blockDim.x + warp_id() * WARP_SZ + w; // Lane 'w' has enlisted entire warp
    for (int idx = leader; idx < n; idx += blockDim.x * gridDim.x) {
      int v = WL[idx];
      int begin = edges[v];
      int end = edges[v+1];
      int degree = end - begin;
      for (int i = lane_id(); i < degree; i += WARP_SZ) {
        int u_idx = begin + i;
        int u = dest[u_idx];
        // printf("[%d-%d]: %d->%d\n", warp_id(), lane_id(), v, u);
        int visited_before = atomicExch(&visited[u], 1);
        if (!visited_before) {
          label[u] = label[v] + 1;
          int next_slot = atomicAdd(&n_out_s, 1);
          if (next_slot >= BLOCK_QUEUE_SIZE) {
            // Block Queue is full
            // printf("overflow\n");
            next_slot = atomicAdd(n_out, 1);
            out_WL[next_slot] = u;
            n_out_s = BLOCK_QUEUE_SIZE;
          } else {
            block_WL[next_slot] = u;
          }
        }
      }
    }
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    block_n_out = atomicAdd(n_out, n_out_s);
    // printf("%d %d\n", block_n_out, n_out_s);
  }
  __syncthreads();

  CUDA_BLOCK_LOOP(idx, n_out_s) {
    out_WL[block_n_out + idx] = block_WL[idx];
  }
  return;
}

__global__ void bfs_gpu_o4(int n, int *WL, int *n_out, int *out_WL, int *edges, int *dest, int *label, int *visited) {
  __shared__ int block_WL[BLOCK_QUEUE_SIZE];
  __shared__ int n_out_s;
  __shared__ int block_n_out;
  int tid = threadIdx.x + blockIdx.x * blockDim.x;

  if (threadIdx.x == 0) n_out_s = 0;
  __syncthreads();

  for (int b = 0; b < blockDim.x; b++) {
    int leader = blockIdx.x * blockDim.x + b; // Lane 'w' has enlisted entire CTA
    for (int idx = leader; idx < n; idx += blockDim.x * gridDim.x) {
      int v = WL[idx];
      int begin = edges[v];
      int end = edges[v+1];
      int degree = end - begin;
      for (int i = threadIdx.x; i < degree; i += blockDim.x) {
        int u_idx = begin + i;
        int u = dest[u_idx];
        // printf("[%d]: %d->%d\n", threadIdx.x, v, u);
        int visited_before = atomicExch(&visited[u], 1);
        if (!visited_before) {
          label[u] = label[v] + 1;
          int next_slot = atomicAdd(&n_out_s, 1);
          if (next_slot >= BLOCK_QUEUE_SIZE) {
            // Block Queue is full
            // printf("overflow\n");
            next_slot = atomicAdd(n_out, 1);
            out_WL[next_slot] = u;
            n_out_s = BLOCK_QUEUE_SIZE;
          } else {
            block_WL[next_slot] = u;
          }
        }
      }
    }
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    block_n_out = atomicAdd(n_out, n_out_s);
    // printf("%d %d\n", block_n_out, n_out_s);
  }
  __syncthreads();

  CUDA_BLOCK_LOOP(idx, n_out_s) {
    out_WL[block_n_out + idx] = block_WL[idx];
  }
  return;
}

typedef struct EdgeItem {
  int indice;
  int parent;
} Edge_t;

#define MAX_EDGE_FRONTIER_SIZE (BLOCK_SZ * 60)

__global__ void bfs_gpu_o5(int n, int *WL, int *n_out, int *out_WL, int *edges, int *dest, int *label, int *visited) {
  typedef cub::BlockScan<int, BLOCK_SZ> BlockScan;
   __shared__ typename BlockScan::TempStorage temp_storage;

  __shared__ int block_WL[BLOCK_QUEUE_SIZE];
  __shared__ int n_out_s;
  __shared__ int block_n_out;
  __shared__ int n_edge_frontier;
  extern __shared__ Edge_t edge_frontier[];

  int degree = 0;
  int indice = 0;
  int tid = threadIdx.x + blockIdx.x * blockDim.x;

  if (threadIdx.x == 0) n_out_s = 0;

  CUDA_KERNEL_LOOP(idx, n) {
    int v = WL[idx];
    degree += edges[v+1] - edges[v];
    // printf("[%d]: degree = %d,\n", tid, degree);
  }
  __syncthreads();

  BlockScan(temp_storage).ExclusiveSum(degree, indice);
  if (threadIdx.x == blockDim.x-1) {
    n_edge_frontier = degree + indice;
    if (n_edge_frontier >= MAX_EDGE_FRONTIER_SIZE) {
      printf("Error [%d]: edge_frontier overflow.\n", tid);
    }
  }
  __syncthreads();
  // printf("[%d]: degree = %d, indice = %d, n_e=%d\n", tid, degree, indice, n_edge_frontier);
  // __syncthreads();

  int start = indice;
  CUDA_KERNEL_LOOP(idx, n) {
    int v = WL[idx];
    for (int i = edges[v]; i < edges[v+1]; i++) {
      edge_frontier[indice].indice = i;
      edge_frontier[indice].parent = v;
      indice++;
    }
    if (indice > degree + start) {
      printf("Error: edge_frontier per thread %d overflow.\n", tid);
      printf("[%d]: degree = %d, indice = %d, start=%d\n", tid, degree, indice, start);
    }
  }
  __syncthreads();

  CUDA_BLOCK_LOOP(idx, n_edge_frontier) {
    int u = dest[edge_frontier[idx].indice];
    int v = edge_frontier[idx].parent;
    // printf("[%d]: %d->%d\n", tid, v, u);
    int visited_before = atomicExch(&visited[u], 1);
    if (!visited_before) {
      label[u] = label[v] + 1;
      int next_slot = atomicAdd(&n_out_s, 1);
      if (next_slot >= BLOCK_QUEUE_SIZE) {
        // Block Queue is full
        // printf("overflow\n");
        next_slot = atomicAdd(n_out, 1);
        out_WL[next_slot] = u;
        n_out_s = BLOCK_QUEUE_SIZE;
      } else {
        block_WL[next_slot] = u;
      }
    }
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    block_n_out = atomicAdd(n_out, n_out_s);
    // printf("%d %d\n", block_n_out, n_out_s);
  }
  __syncthreads();

  CUDA_BLOCK_LOOP(idx, n_out_s) {
    out_WL[block_n_out + idx] = block_WL[idx];
  }
}

#define bfs_gpu_ox bfs_gpu_o4

TEST(BFS, Gpu_OX) {
  int n = 150000;
  int e = 750000;
  MemGraph Gx;
  initMemGraph(Gx, n, e);
  CSRGraph G;
  G.edges = new int[n+1];
  G.dest = new int[e];
  int *label = new int[n];
  memset(label, 0, n *sizeof(int));
  int *visited = new int[n];
  memset(visited, 0, n *sizeof(int));

  convert(n, e, Gx, G);
  // printGraph(G);
  int src = 0;
  visited[src] = 1;

  device_vector<int> edges_d(G.edges, G.edges+n+1);
  device_vector<int> dest_d(G.dest, G.dest+e);
  device_vector<int> label_d(label, label+n);
  device_vector<int> visited_d(visited, visited+n);
  device_vector<int> WL_in(e, 0);
  device_vector<int> WL_out(e, 0);
  device_vector<int> n_out(1, 0);
  int n_wl = 1;
  WL_in[0] = src;
  int *wl_in = PTR(WL_in);
  int *wl_out = PTR(WL_out);
cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);
cudaEventRecord(start);
  while (n_wl > 0) {
    int gridSize = (n_wl + BLOCK_SZ - 1) / BLOCK_SZ;
    bfs_gpu_ox <<<gridSize, BLOCK_SZ>>> (n_wl, wl_in, PTR(n_out), wl_out, PTR(edges_d), PTR(dest_d), PTR(label_d), PTR(visited_d));
    // std::cout << "after loop: \n";
    n_wl = n_out[0];
    EXPECT_LE(n_wl, WL_out.size());
    n_out[0] = 0;
    int *tmp = wl_in; wl_in = wl_out; wl_out = tmp;
    // std::cout << "WL=" << n_wl << std::endl;
    // printCudaArray(n, PTR(label_d));
    // break;
  }
cudaEventRecord(stop);
cudaEventSynchronize(stop);
float milliseconds = 0;
cudaEventElapsedTime(&milliseconds, start, stop);
std::cout << "Time: " << milliseconds << " (ms)." << endl;

  bfs_cpu(G, src, label);
  // printArray(n, label);

  for (int i = 0; i < n; i++) {
    // std::cout << i << endl;
    CHECK_EQ(label[i], label_d[i]);
  }
  delete[] G.edges;
  delete[] G.dest;
  delete[] label;
  delete[] visited;
}

TEST(BFS, Gpu_O5) {
  int n = 150000;
  int e = 750000;
  MemGraph Gx;
  initMemGraph(Gx, n, e);
  CSRGraph G;
  G.edges = new int[n+1];
  G.dest = new int[e];
  int *label = new int[n];
  memset(label, 0, n *sizeof(int));
  int *visited = new int[n];
  memset(visited, 0, n *sizeof(int));

  convert(n, e, Gx, G);
  // printGraph(G);
  int src = 0;
  visited[src] = 1;

  device_vector<int> edges_d(G.edges, G.edges+n+1);
  device_vector<int> dest_d(G.dest, G.dest+e);
  device_vector<int> label_d(label, label+n);
  device_vector<int> visited_d(visited, visited+n);
  device_vector<int> WL_in(e, 0);
  device_vector<int> WL_out(e, 0);
  device_vector<int> n_out(1, 0);
  int n_wl = 1;
  WL_in[0] = src;
  int *wl_in = PTR(WL_in);
  int *wl_out = PTR(WL_out);
cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);
cudaEventRecord(start);
  while (n_wl > 0) {
    int gridSize = (n_wl + BLOCK_SZ - 1) / BLOCK_SZ;
    bfs_gpu_o5 <<<gridSize, BLOCK_SZ, MAX_EDGE_FRONTIER_SIZE*sizeof(Edge_t)>>> (n_wl, wl_in, PTR(n_out), wl_out, PTR(edges_d), PTR(dest_d), PTR(label_d), PTR(visited_d));
    CUDA_CHECK( cudaGetLastError() );
    n_wl = n_out[0];
    EXPECT_LE(n_wl, WL_out.size());
    n_out[0] = 0;
    int *tmp = wl_in; wl_in = wl_out; wl_out = tmp;
    // std::cout << "loop WL=" << n_wl << std::endl;
    // printCudaArray(n, PTR(label_d));
    // break;
  }
cudaEventRecord(stop);
cudaEventSynchronize(stop);
float milliseconds = 0;
cudaEventElapsedTime(&milliseconds, start, stop);
std::cout << "Time: " << milliseconds << " (ms)." << endl;

  bfs_cpu(G, src, label);
  // printArray(n, label);

  for (int i = 0; i < n; i++) {
    // std::cout << i << endl;
    CHECK_EQ(label[i], label_d[i]) << " at node " << i;
  }
  delete[] G.edges;
  delete[] G.dest;
  delete[] label;
  delete[] visited;
}

