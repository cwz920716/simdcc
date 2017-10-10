__task__ void init(Vertex[] Vertices) {
  __device__ parfor(auto v : Vertices) {
    v.dist = INT_MAX;
  }
}

__task__ void warp_gather(Vertex[] Vertices) {
  __device__ parfor(auto v : Vertices) {
    __thread__ Vertex[] adj = v.neighbours;
    __thread__ bool adj_valid = adj.size > 0;
    __warp__ enlist(adj_valid, adj -> arg0) {
      __warp__ parfor(v : leader.arg0) {
        visit(v);
      }
    }
  }
}

__task__ void warp_gather_detail(Vertex[] Vertices) {
  __device__ parfor(auto v : Vertices) {
    __warp__ Leader[2];
    __thread__ Vertex[] adj = v.neighbours;
    __thread__ bool adj_valid = adj.size > 0;
    while (__warp__ any(adj_valid)) {
      if (adj_valid) Leader[0] = lane_id;
      if (Leader[0] == lane_id) {
        Leader[1] = v;
      }
      __warp__ parfor (w : Leader[1]) {
        visit(w);
      }
    }
  }
}
 
__task__ void fg_gather(Vertex[] Vertices) {
  __device__ parfor(auto v : Vertices) {
    __threadblock__ int comm[THREADS_PER_BLOCK];
    __thread__ int tb_progress = 0;
    __thread__ int total;
    __thread__ int rsv_rank;
    __thread__ Vertex[] adj = v.neighbours;
    rsv_rank, total = __threadblock__ ExclusiveScan(adj.size);
    __thread__ int iter = adj.begin();
    __uniform__ while(tb_progress < total) {
      while (rsv_rank - tb_progress < comm.capacity && iter < adj.end()) {
        comm[rsv_rank] = adj[iter];
        iter++;
        rsv_rank++;
      }
      __threadblock__ barrier();

      __threadblock__ parfor(v : comm) {
        visit(v);
      }

      tb_progress += comm.capacity;
      __threadblock__ barrier();  // Is this barrier removable?
    }
  }
}

__task__ void fg_gather_advanced(Vertex[] Vertices) {
  __device__ parfor(auto v : Vertices) {
    __threadblock__ __use_once__ WorkList<int, THREADS_PER_BLOCK> comm;
    __threadblock__ int tb_progress = 0;
    __threadblock__ int total;
    __thread__ int rsv_rank;
    __thread__ Vertex[] adj = v.neighbours;
    __threadblock__ ResourceAllocation(comm, adj);
    __threadblock__ parfor(v : comm) {
      visit(v);
    }
  }
}

__task__ void spv_sum(float[] C, float[] A) {
  __device__ Slice rows = [0:C.size:1];
  __device__ parfor(i : rows) {
    __thread__ Slice cols = [ C[i] : C[i+1] : 1 ];
    __thread__ sum = 0;
    for (j : cols) {
      sum += A[j];
    }
  }
}

__task__ void warp_cull(Vertex[] Vertices) {
  __device__ parfor(v : Vertices) {
    __thread__ int hash = v & 127;
    __warp__ scratch[128];
    scratch[hash] = v;
    if (scratch[hash] == v) {
      scratch[hash] = thread_id;
      if (scratch[hash] != thread_id) {
        v.duplicate = true;
      }
    }
    v.duplicate = false; // there might be false positives
  }
}
