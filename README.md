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
 {   
fn = "apu_float32_1_1"    
f = h5py.File(fn, 'w')

for k, v in attrs.items():
        f.attrs[k] = v

times = f.create_dataset('times', (len(timed),), 'f')
    
neighbors = f.create_dataset('neighbors', (len(neigh), 10), 'i')
    
distances = f.create_dataset('distances', (len(dist), 10), 'f')

times = timed
neighbors = neigh
distances = dist
             
f.close()
}

These parameters will have to be adjusted according to the dataset you are benchmarking. 
