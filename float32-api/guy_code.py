import sys
import gsi_float_neural_api

print("---------------")
print("START SCRIPT")
print("---------------")

float32_host = "localhost"
dataset_path = "/efs/data/qa/any_vision/dataset.npy"
queries_path = "/efs/data/qa/any_vision/queries.npy"
num_of_boards = 1
neural_matrix_path = "/efs/data/public/Daphna/AnyVision_NPHash_Weight_Files/any_vision_1589923812.533218__dim256_loss7_v3random_proxies_nlFalse_hns256.0_scale30_optimizerAdam_lr0.001_proxies5000_trainFalse_hidden2_tss100000000_merged_weights.npy"


gsi_conf = gsi_float_neural_api.GSIConfig(log_level="INFO", float32_host=float32_host)
gsi = gsi_float_neural_api.Float_GSI(conf=gsi_conf)
gsi.delete_allocation('4cfb317a-f903-11ea-92f3-0242ac110010')

print("---------------")
print("IMPORTING DATA .....")
print("---------------")

dataset_id, status = gsi.import_dataset(dataset_path)
if status != "OK":
    print(f"import_dataset failed with the following error: {status}")
    sys.exit(1)

    
allocation_id, status = gsi.request_allocation(num_of_boards)
if status != "OK":
    print(f"request_allocation failed with the following error: {status}")
    sys.exit(1)

    
status = gsi.load_data(allocation_id, dataset_id, neural_matrix_path, hamming_k=3300, normalize=False, typical_nqueries=50, max_nqueries=3000, topk=25)
if status != "OK":
    print(f"load_data failed with the following error: {status}")
    sys.exit(1)

print("---------------")
print("STARTING SEARCH ....")
print("---------------")

search = gsi.Search()

distance, indices, search_time, status = search.knn_composite_cosine(allocation_id, dataset_id, queries_path)
if status != "OK":
    print(f"search.knn_composite_cosine failed with the following error: {status}")
    sys.exit(1)
search_result = list(zip(indices, distance))
print(f"result is {str(search_result)}")
print(f"search time = {str(format(search_time, '.6f'))}")



status = gsi.delete_allocation(allocation_id)
if status != "OK":
    print(f"delete_allocation failed with the following error: {status}")
    sys.exit(1)

print("---------------")
print("SCRIPT DONE")
print("---------------")