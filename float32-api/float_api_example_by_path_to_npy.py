import sys
import gsi_float_api
import numpy as np
import csv
from datetime import datetime


def search_e2e(gsi, num_of_boards, dataset_id, queries_path):
    allocation_id, status = gsi.request_allocation(num_of_boards)
    if status != "OK":
        print(f"request_allocation failed with the following error: {status}")
        sys.exit(1)

    np_queries = np.load(queries_path)
    status = gsi.load_data(allocation_id, dataset_id, typical_nqueries=np_queries.shape[0], max_nqueries=3000)
    if status != "OK":
        print(f"load_data failed with the following error: {status}")
        sys.exit(1)

    search = gsi.Search()

    num_of_iter = 100
    if np_queries.shape[0] > 10:
        num_of_iter = 10

    total_search_time = 0
    for i in range(num_of_iter):
        distance, indices, search_time, status = search.knn_composite_cosine(allocation_id, dataset_id, queries_path,
                                                                             topk=25)
        if status != "OK":
            print(f"search.knn_composite_cosine failed with the following error: {status}")
            sys.exit(1)
        total_search_time = total_search_time + search_time

    average_search_time = total_search_time / num_of_iter

    status = gsi.unload_data(allocation_id, dataset_id)
    if status != "OK":
        print(f"unload_data failed with the following error: {status}")
        sys.exit(1)

    status = gsi.delete_allocation(allocation_id)
    if status != "OK":
        print(f"delete_allocation failed with the following error: {status}")
        sys.exit(1)

    return average_search_time


def search_multiple(num_boards, dataset):
    resultt_time_1 = search_e2e(gsi_api, num_boards, dataset, "/efs/data/qa/any_vision/queries_1.npy")
    resultt_time_10 = search_e2e(gsi_api, num_boards, dataset, "/efs/data/qa/any_vision/queries_10.npy")
    resultt_time_50 = search_e2e(gsi_api, num_boards, dataset, "/efs/data/qa/any_vision/queries_50.npy")
    resultt_time_100 = search_e2e(gsi_api, num_boards, dataset, "/efs/data/qa/any_vision/queries_100.npy")
    resultt_time_1000 = search_e2e(gsi_api, num_boards, dataset, "/efs/data/qa/any_vision/queries_1000.npy")

    return resultt_time_1, resultt_time_10, resultt_time_50, resultt_time_100, resultt_time_1000


def search_single(num_boards, dataset, queries):
    res_single = search_e2e(gsi_api, num_boards, dataset, queries)
    return res_single


gsi_conf = gsi_float_api.GSIConfig(log_level="INFO", float32_host="localhost")
gsi_api = gsi_float_api.Float_GSI(conf=gsi_conf)

dataset_128 = "f4caaf3e-cf50-11ea-9d42-6affbdf66ebc"
dataset_256 = "723483c6-d4d5-11ea-87a0-6affbdf66ebc"
dataset_384 = "a9fe27c6-d4d5-11ea-87a0-6affbdf66ebc"
dataset_500 = "174bd87a-cce6-11ea-bf45-6affbdf66ebc"
dataset_1M = "be524030-d4d6-11ea-87a0-6affbdf66ebc"
dataset_5M = "1c2fbfe2-d4d8-11ea-87a0-6affbdf66ebc"
dataset_8M = "8d8ebe20-d4da-11ea-87a0-6affbdf66ebc"

file_path = "/home/public/data/float_results/" + "results_" + str(datetime.now().strftime("%Y%m%d%H%M%S")) + ".csv"
with open(file_path, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["DB-Size", "Num Of APUs", "1 query", "10 queries", "50 queries", "100 queries", "1000 queries"])

    result_time_1, result_time_10, result_time_50, result_time_100, result_time_1000 = search_multiple(1, dataset_128)
    writer.writerow(["128k-search", "1", str(result_time_1), str(result_time_10), str(result_time_50),
                     str(result_time_100), str(result_time_1000)])

    result_time_1, result_time_10, result_time_50, result_time_100, result_time_1000 = search_multiple(2, dataset_256)
    writer.writerow(["256k", "2", str(result_time_1), str(result_time_10), str(result_time_50),
                     str(result_time_100), str(result_time_1000)])

    result_time_1, result_time_10, result_time_50, result_time_100, result_time_1000 = search_multiple(3, dataset_384)
    writer.writerow(["384k", "3", str(result_time_1), str(result_time_10), str(result_time_50),
                     str(result_time_100), str(result_time_1000)])

    #####################################################

    writer.writerow(["DB-Size", "Num Of APUs", "1 query", "10 queries", "50 queries", "100 queries", "1000 queries"])

    result_time_1, result_time_10, result_time_50, result_time_100, result_time_1000 = search_multiple(1, dataset_500)
    writer.writerow(["500k", "1", str(result_time_1), str(result_time_10), str(result_time_50),
                     str(result_time_100), str(result_time_1000)])

    result_time_1, result_time_10, result_time_50, result_time_100, result_time_1000 = search_multiple(1, dataset_1M)
    writer.writerow(["1M", "1", str(result_time_1), str(result_time_10), str(result_time_50),
                     str(result_time_100), str(result_time_1000)])

    result_time_1, result_time_10, result_time_50, result_time_100, result_time_1000 = search_multiple(1, dataset_5M)
    writer.writerow(["5M", "1", str(result_time_1), str(result_time_10), str(result_time_50),
                     str(result_time_100), str(result_time_1000)])

    result_time_1, result_time_10, result_time_50, result_time_100, result_time_1000 = search_multiple(1, dataset_8M)
    writer.writerow(["8M", "1", str(result_time_1), str(result_time_10), str(result_time_50),
                     str(result_time_100), str(result_time_1000)])

    #####################################################

    writer.writerow(["DB-Size", "Num Of APUs", "1 query", "10 queries", "50 queries", "100 queries", "1000 queries"])

    result_time_1, result_time_10, result_time_50, result_time_100, result_time_1000 = search_multiple(2, dataset_500)
    writer.writerow(["500k", "2", str(result_time_1), str(result_time_10), str(result_time_50),
                     str(result_time_100), str(result_time_1000)])

    result_time_1, result_time_10, result_time_50, result_time_100, result_time_1000 = search_multiple(2, dataset_1M)
    writer.writerow(["1M", "2", str(result_time_1), str(result_time_10), str(result_time_50),
                     str(result_time_100), str(result_time_1000)])

    result_time_1, result_time_10, result_time_50, result_time_100, result_time_1000 = search_multiple(2, dataset_5M)
    writer.writerow(["5M", "2", str(result_time_1), str(result_time_10), str(result_time_50),
                     str(result_time_100), str(result_time_1000)])

    result_time_1, result_time_10, result_time_50, result_time_100, result_time_1000 = search_multiple(2, dataset_8M)
    writer.writerow(["8M", "2", str(result_time_1), str(result_time_10), str(result_time_50),
                     str(result_time_100), str(result_time_1000)])

    #####################################################
    writer.writerow(["DB-Size", "Num Of APUs", "1 query", "10 queries", "50 queries", "100 queries", "1000 queries"])

    result_time_1, result_time_10, result_time_50, result_time_100, result_time_1000 = search_multiple(3, dataset_500)
    writer.writerow(["500k", "3", str(result_time_1), str(result_time_10), str(result_time_50),
                     str(result_time_100), str(result_time_1000)])

    result_time_1, result_time_10, result_time_50, result_time_100, result_time_1000 = search_multiple(3, dataset_1M)
    writer.writerow(["1M", "3", str(result_time_1), str(result_time_10), str(result_time_50),
                     str(result_time_100), str(result_time_1000)])

    result_time_1, result_time_10, result_time_50, result_time_100, result_time_1000 = search_multiple(3, dataset_5M)
    writer.writerow(["5M", "3", str(result_time_1), str(result_time_10), str(result_time_50),
                     str(result_time_100), str(result_time_1000)])

    result_time_1, result_time_10, result_time_50, result_time_100, result_time_1000 = search_multiple(3, dataset_8M)
    writer.writerow(["8M", "3", str(result_time_1), str(result_time_10), str(result_time_50),
                     str(result_time_100), str(result_time_1000)])
