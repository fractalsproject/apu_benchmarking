#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import sys
import gsi_float_neural_api
import numpy as np
import csv
from datetime import datetime
import h5py


# ### Allocate Boards

# Allocate boards on the correct localhost, don't proceed until status is "OK"

# In[ ]:


gsi_conf = gsi_float_neural_api.GSIConfig()
print(gsi_conf)
gsi = gsi_float_neural_api.Float_GSI(conf=gsi_conf)


# In[ ]:


#gsi.delete_allocation('5b51814e-fd12-11ea-b7c7-0242ac110014')


# In[ ]:


alloc_id, status = gsi.request_allocation(2) #Only 2 right now

print("Allocation ID: " + str(alloc_id))
print("Status: " + str(status))


# ### Load Data

# Loading dataset from server since it is mounted there, loading query set from local npy file

# In[ ]:


#dataset_id, result = gsi.import_dataset("/efs/data/qa/any_vision/dataset.npy")
dataset_id, result = gsi.import_dataset("/efs/data/qa/any_vision/deep-image-96.npy")
print("Dataset ID: " + str(dataset_id))
print("Status: " + str(result))
neural_matrix_path = "/efs/data/public/Daphna/AnyVision_NPHash_Weight_Files/any_vision_1589923812.533218__dim256_loss7_v3random_proxies_nlFalse_hns256.0_scale30_optimizerAdam_lr0.001_proxies5000_trainFalse_hidden2_tss100000000_merged_weights.npy"


# In[ ]:


#create the queryset
#fl = np.load('queries.npy')
fl = np.load("/efs/data/qa/any_vision/deep1b_queries.npy")

#testing query's working
#search = gsi.Search()

#fp = np.load('/efs/data/qa/any_vision/queries_1000.npy')
#status = gsi.load_data(alloc_id, dataset_id, neural_matrix_path, hamming_k=100, normalize=False, typical_nqueries=1000, max_nqueries=1000)
#distance, indices, search_time, status = search.knn_composite_cosine_by_queries(alloc_id, dataset_id, fp)


# ### Search

# Search using different Hamming values

# In[ ]:


search = gsi.Search()

ham = 10
time_list = []

total_search_time = 0
print("Search Times (QPS) for GSI_APU neural float32 Search:")
print("----------------------------------------------")
print()
print("RUNNING ...")

timed = []
neigh = []
dist = []

print("About to load data...")
status = gsi.load_data(alloc_id, dataset_id, neural_matrix_path, hamming_k=ham, normalize=False, typical_nqueries=1, max_nqueries=1)
print("Done loading data...")
print(status)
total_search = 0
for i in range(len(fl)):
    
    print("About to do a query...")
    distance, indices, search_time, status = search.knn_composite_cosine_by_queries(alloc_id, dataset_id, fl[i])
    print("Done with the query...")
    
    timed.append(search_time)
    neigh.append(indices)
    dist.append(distance)
    
    break # let's just try 1 right now
    
    if i % 1000 == 0:
        print(i)
    
    total_search = total_search + search_time

print("----------------------------------------------")
print("Total Search Time: " +str(total_search))
print("Hamming_k Value of: " + str(ham),end = " runs at ")
print(str(np.round(1.0/(total_search/10000),2)) + " Average Queries per Second")


# In[ ]:


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
        'dataset': 'nytimes-256-angular'
    }


# In[ ]:


#SAVE IN ANN_BENCH FORMAT
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


# In[ ]:


status = gsi.unload_data(alloc_id, dataset_id)
status = gsi.delete_allocation(alloc_id)


# In[ ]:





# In[ ]:





# ## TESTING FORMAT, not relevent

# In[ ]:


check = h5py.File("angular_hnsw_M_12_efConstruction_800_post_0_false_1","r")
check.keys()


# In[ ]:


list(check[list(check.keys())[1]])[:100]

