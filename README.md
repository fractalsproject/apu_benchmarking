## APU Benchmarking
This repo contains code for testing and benchmarking APU#2’s float32 search, before formatting it into a file time applicable for Erik Bernhardsson’s ANN-Benchmarks.



Neural_tester: This is the notebook for testing the float32-NN powered search. For this to work the NN must be pretrained on the dataset. Currently this has only been done from the anyvision set.

Faiss_tester: This is the notebook for testing float32 as powered by FAISS. This works on any ds but is much less stable.

Both notebooks use the knn-search by query, hence a query set has to be created locally. If you are using ANN benchmarks this is done automatically when creating a dataset. Hence I recommend downloading Erik's ds and converting them from hdf5 to a ds and query set for testing. After the benchmark has finished it is saved in the following format:


attrs = {

        "algo": "float32_1board",
        
        "batch_mode": False,
        
        "best_search_time": total_search/10000,
        
        "candidates": 1,
        
        "expect_extra": False,
        
        "name": "float32",
        
        "run_count": 2,
        
        "distance": "angular",
        
        "count": 10,
        
        'dataset': 'deep-image-96-angular'
        
    }

These parameters will have to be adjusted according to the dataset you are benchmarking. 

WARNING: as of 09/25/20 trying to benchmark on 1% of deep1b will result in a dataset size related error (/usr/local/bin/docker-entrypoint.sh: line 3:     6 Segmentation fault      (core dumped) ./server)
