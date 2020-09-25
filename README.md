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



Plotting: To plot the benchmark files move them into Erik's repo results/dataset/10 from ANN-benchmarking and create a seperate folder called float32. For example if you were benchmarking on deep1b you would navigate to ann-benchmarks/results/deep-image-96-angular/10, here you would see a list of folders for each algo. Create your own folder and place the newly created hdf5 float32 benchmarking files inside. Then run the plotting algorithm from Erik's project as you normally would.

Because deep-image-96-angular is too large for the current apu set up I have benchmarked glOVe-50-angular. These files are ready for compairsion to apu#2-float32 you just have to take the glove-50-angular dataset and convert it to npy format.
