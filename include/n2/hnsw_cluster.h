
#pragma once
/** @file */
#include <omp.h>

#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "thread_pool.h"
#include "hnsw.h"

namespace n2 {

class HnswCluster {
public:

    HnswCluster(int thread_pool_size, int dim, std::vector<std::string> hnsw_paths, std::string metric);

    ~HnswCluster();

    void wait_end();
    
    void SearchByVector(const std::vector<float>& qvec, size_t k, 
                               size_t ef_search,
                               std::vector<std::vector<std::pair<int, float>>>& result);
private:
    std::vector<Hnsw*> cluster;
    ctpl::thread_pool *pool;
    std::vector<std::pair<int,float>>* results;
    int max_threads;

};


}