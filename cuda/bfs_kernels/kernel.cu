/*********************************************************************************
Implementing Breadth first search on CUDA using algorithm given in HiPC'07
  paper "Accelerating Large Graph Algorithms on the GPU using CUDA"

Copyright (c) 2008 International Institute of Information Technology -
Hyderabad.
All rights reserved.

Permission to use, copy, modify and distribute this software and its
documentation for
educational purpose is hereby granted without fee, provided that the above
copyright
notice and this permission notice appear in all copies of this software and that
you do
not sell the software.

THE SOFTWARE IS PROVIDED "AS IS" AND WITHOUT WARRANTY OF ANY KIND,EXPRESS,
IMPLIED OR
OTHERWISE.

The CUDA Kernel for Applying BFS on a loaded Graph. Created By Pawan Harish
**********************************************************************************/
#ifndef _KERNEL_H_
#define _KERNEL_H_

__device__ void clock_block(clock_t clock_count)
{
    clock_t start_clock = clock();
    clock_t clock_offset = 0;
    while (clock_offset < clock_count)
    {
        clock_offset = clock() - start_clock;
    }
}

__global__ void Kernel(Node *g_graph_nodes, int *g_graph_edges,
                       bool *g_graph_mask, bool *g_updating_graph_mask,
                       bool *g_graph_visited, int *g_cost, int no_of_nodes) {
    int tid = blockIdx.x * MAX_THREADS_PER_BLOCK + threadIdx.x;
    if (tid < no_of_nodes && g_graph_mask[tid]) {
        clock_block(100);
        g_graph_mask[tid] = false;
        for (int i = g_graph_nodes[tid].starting;
             i < (g_graph_nodes[tid].no_of_edges + g_graph_nodes[tid].starting);
             i++) {
            clock_block(1e4);
            int id = g_graph_edges[i];
            if (!g_graph_visited[id]) {
                g_cost[id] = g_cost[tid] + 1;
                g_updating_graph_mask[id] = true;
            }
        }
    }
}

__global__ void Kernel_ir(Node *g_graph_nodes, int *g_graph_edges,
                           bool *g_graph_mask, bool *g_updating_graph_mask,
                           bool *g_graph_visited, int *g_cost, int no_of_nodes) {
    __shared__ int task_q[MAX_THREADS_PER_BLOCK];
    __shared__ int head;

    if (threadIdx.x == 0) {
      head = 0;
    }
    __syncthreads();

    int tid = blockIdx.x * MAX_THREADS_PER_BLOCK + threadIdx.x;
    bool cond = (tid < no_of_nodes && g_graph_mask[tid]);
    int loc = 0;
    if (cond) {
      loc = atomicAdd(&head, 1);
      task_q[loc] = tid;
    }
    __syncthreads();

    if (threadIdx.x < head) {
        tid = task_q[threadIdx.x];
        g_graph_mask[tid] = false;
        for (int i = g_graph_nodes[tid].starting;
             i < (g_graph_nodes[tid].no_of_edges + g_graph_nodes[tid].starting);
             i++) {
            clock_block(1e4);
            int id = g_graph_edges[i];
            if (!g_graph_visited[id]) {
                g_cost[id] = g_cost[tid] + 1;
                g_updating_graph_mask[id] = true;
            }
        }
    }
}

#endif
