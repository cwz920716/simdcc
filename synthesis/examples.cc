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

__task__ void fg_gather(Vertex[] Vertices) {
  __device__ parfor(auto v : Vertices) {
    __threadblock__ int comm[THREADS_PER_BLOCK];
    __threadblock__ int tb_progress = 0;
    __threadblock__ int total;
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
