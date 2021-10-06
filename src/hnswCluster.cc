#include "n2/hnsw_cluster.h"
#include <iostream>

#pragma once
/** @file */
#include <omp.h>

namespace n2 {

    void HnswCluster::wait_end(){

        pool->stop(true);
        delete pool;
        pool=new ctpl::thread_pool(max_threads);
    }

    HnswCluster::HnswCluster(int thread_pool_size, int dim,
                             std::vector<std::string> hnsw_paths, std::string metric="angular") {
        pool = new ctpl::thread_pool(thread_pool_size);
        max_threads = thread_pool_size;
        results = new std::vector<std::pair<int,float>>[hnsw_paths.size()];
        for(int i=0;i<(int)hnsw_paths.size();i++){
            Hnsw* cur = new Hnsw(dim, metric);
            cur->LoadModel(hnsw_paths[i], true);
            cluster.push_back(cur);
        }
    }

    HnswCluster::~HnswCluster(){
        delete pool;
        delete results;
        for (int i = 0; i < (int)cluster.size(); i++)
        {
            delete cluster[i];
        }
    }

    bool cmp(std::pair<int, float> x, std::pair<int,float> y){
        return x.second < y.second;
    }

    void search_hnsw(int tid, Hnsw* hnsw, const std::vector<float>& qvec,size_t k, size_t ef_search, std::vector<std::pair<int, float>>* result){
        std::vector<std::pair<int, float>> temp;
        hnsw->SearchByVector(qvec, k, ef_search, temp);
        *(result) = temp;
        // delete temp;
        // std::cout << result->size() << '\n';
    }

    void HnswCluster::SearchByVector(const std::vector<float>& qvec, size_t k, 
                               size_t ef_search,
                               std::vector<std::vector<std::pair<int, float>>>& result) {
        // std::vector<std::vector<std::pair<int, float>>> tmp((int)cluster.size());
        // for(int i=0;i<(int)cluster.size();i++){
        //     // results[i].clear();
        //     pool->push(search_hnsw, cluster[i], qvec, k, ef_search, &(tmp[i]));
        // }
        // this->wait_end();
        // result = tmp;
        result.resize(cluster.size());
        #pragma omp parallel num_threads(max_threads)
        {
            #pragma omp for schedule(runtime)
            for (size_t i = 0; i < cluster.size(); ++i) {
                cluster[i]->SearchByVector(qvec, k, ef_search, result[i]);
                // s->SearchByVector(qvecs[i], k, ef_search, ensure_k_, results[i]);
            }
        }
    }

}