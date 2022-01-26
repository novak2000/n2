#include "n2/hnsw_cluster.h"
#include <iostream>
#include "n2/hnsw.h"
using std::runtime_error;

#pragma once
/** @file */
#include <omp.h>

namespace n2 {


    HnswCluster::HnswCluster(int thread_pool_size, int dim,
                             std::vector<std::string> hnsw_paths, std::string metric="angular") {
        max_threads = thread_pool_size;
        data_dim_ = dim;
        // std::vector<std::shared_ptr<HnswSearch>> searcher_pool_;
        
        
        if (metric == "L2" || metric =="euclidean") {
            metric_ = DistanceKind::L2;
        } else if (metric == "angular") {
            metric_ = DistanceKind::ANGULAR;
        } else if (metric == "dot") {
            metric_ = DistanceKind::DOT;
        } else {
            throw runtime_error("[Error] Invalid configuration value for DistanceMethod: " + metric);
        }

        for(int i=0;i<(int)hnsw_paths.size();i++){
            cluster.push_back( HnswModel::LoadModelFromFile(hnsw_paths[i]));
            if (data_dim_ > 0 && (int)data_dim_ != cluster.back()->GetDataDim()) {
                throw runtime_error("[Error] index dimension(" + std::to_string((int)data_dim_) \
                                    + ") != model dimension(" + std::to_string(cluster.back()->GetDataDim()) + ")");
            }
            // TODO: dodati proveru da li su metrike iste
            // if (metric_ != cluster.back()->GetMetric()) {
            //     throw runtime_error("[Error] index dimension(" + to_string(data_dim_)
            //                         + ") != model dimension(" + to_string(cluster.back()->GetDataDim()) + ")");
            // }
            data_dim_ = cluster.back()->GetDataDim();
            metric_ = cluster.back()->GetMetric();
        }
        for(size_t i=0;i<cluster.size();i++)
        {
            searcher_pool_global.push_back(HnswSearch::GenerateSearcher(cluster[i], data_dim_, metric_));
        }
    }

    HnswCluster::~HnswCluster(){
    }


    void HnswCluster::SearchByVector(const std::vector<float>& qvec, size_t k, 
                               size_t ef_search,
                               std::vector<std::vector<std::pair<int, float>>>& result) {
        result.resize(cluster.size());
        // std::vector<std::shared_ptr<HnswSearch>> searcher_pool_;
        // for(size_t i=0;i<cluster.size();i++)
        // {
        //     searcher_pool_.push_back(HnswSearch::GenerateSearcher(cluster[i], data_dim_, metric_));
        // }

        #pragma omp parallel num_threads(cluster.size())
        {
            // #pragma omp for schedule(runtime)
            // for(size_t i = 0; i<cluster.size();i++){
                // if(i==0)
                //     searcher_pool_global[omp_get_thread_num()] = HnswSearch::GenerateSearcher(cluster[omp_get_thread_num()], data_dim_, metric_);
                // auto& s = searcher_pool_global[omp_get_thread_num()];
                auto& s = searcher_pool_global[omp_get_thread_num()];
                s->SearchByVector(qvec, k, ef_search, ensure_k_, result[omp_get_thread_num()]);
            // }
        }
    }

}