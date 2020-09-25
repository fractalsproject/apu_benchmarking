import gsi_api
import gsi_float_api
import numpy as np
import sys
import json

conf = gsi_api.GSIConfig(log_level="DEBUG", sm_host="localhost")
gsi = gsi_float_api.Float_GSI(conf=conf)
dataset_path = "/efs/data/public/trefaeli/faiss_datasets/original.npy"
dataset_id, res = gsi.import_dataset(dataset_path)
if res != "OK":
    print(f"import_dataset failed: {res}")
    sys.exit(1)
alloc_id, res = gsi.request_allocation(1)
if res != "OK":
    print(f"request_allocation failed: {res}")
    sys.exit(1)
dict = {'dataset_id': dataset_id, 'alloc_id': alloc_id}
res = gsi.load_data(alloc_id, dataset_id)
if res != "OK":
    print(f"load_data failed: {res}")
    sys.exit(1)

search = gsi.Search()
queries_path = "/efs/data/public/dsosnovsky/Documents/any2_refine_add_remove/queries_0.npy"
distance, index, time, res = search.knn_composite_cosine(alloc_id,
                                                         dataset_id,
                                                         5,
                                                         queries_path)
if res != "OK":
    print(f"knn_composite_cosine failed: {res}")
search_result = list(zip(index, distance))
print(f"result is {str(search_result)}")
print(f"search time = {str(format(time, '.4f'))}")

last_index, res = gsi.add_data(dataset_id, "/efs/data/public/trefaeli/faiss_datasets/queries_0.npy")
if res != "OK":
    print(f"add_data failed: {res}")
    sys.exit(1)

res = gsi.remove_data(dataset_id, [last_index])
if res != "OK":
    print(f"remove_data failed: {res}")
    sys.exit(1)

res = gsi.unload_data(alloc_id, dataset_id)
if res != "OK":
    print(f"unload_data failed: {res}")
    sys.exit(1)
    
res = gsi.delete_allocation(alloc_id)
if res != "OK":
    print(f"delete_allocation failed: {res}")
    sys.exit(1)