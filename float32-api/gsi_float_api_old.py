import gsi_api
from gsi_api import exception_handling, exception_handling_void, check_response, GSIConfig, StopWatch, SHORT_REQUEST_TIMEOUT_SECS, LONG_REQUEST_TIMEOUT_SECS
import requests
import numpy as np
from datetime import datetime
import logging
        
def validate_top_k(top_k):
    if type(top_k) != int:
        raise Exception("top_k must be int")

class Float_GSI(gsi_api.GSI):
    def __init__(self, conf=GSIConfig()):
        super().__init__(conf)
        
    @exception_handling
    def request_allocation(self, num_boards, name=None):
        gsi_api.validate_num_boards(num_boards)
        request = {
            "name":name,
            "numOfBoards": num_boards
        }
        stopwatch = StopWatch(name="request_allocation")
        response = requests.post(f"{self.conf.gateway_base_url}/float/resource/allocation/_request",
                                 json=request,
                                 timeout=SHORT_REQUEST_TIMEOUT_SECS)
        check_response(response)
        if response.json()["allocId"] is None:
            raise Exception("ERR_NO_MORE_ALLOC")
        stopwatch.print_elapsed_time()
        return response.json()["allocId"]

    @exception_handling_void
    def delete_allocation(self, alloc_id):
        stopwatch = StopWatch(name="delete_allocation")
        response = requests.delete(f"{self.conf.gateway_base_url}/float/resource/allocation/{alloc_id}",
                                 timeout=SHORT_REQUEST_TIMEOUT_SECS)
        check_response(response)
        stopwatch.print_elapsed_time()

    @exception_handling
    def import_dataset(self, dataset_path, name=None):
        request = {
            "name": name,
            "filePath": dataset_path
        }

        stopwatch = StopWatch()
        response = requests.post(f"{self.conf.gateway_base_url}/float/dataset/_import",
                                 json=request,
                                 timeout=SHORT_REQUEST_TIMEOUT_SECS)
        check_response(response)
        stopwatch.print_elapsed_time()
        return response.json()["datasetId"]
        
    @exception_handling_void
    def load_data(self, alloc_id, dataset_id):
        request = {
            "datasetId": dataset_id,
            "allocId": alloc_id
        }

        stopwatch = StopWatch()
        response = requests.post(f"{self.conf.gateway_base_url}/float/dataset/_load",
                                 json=request,
                                 timeout=LONG_REQUEST_TIMEOUT_SECS)
        check_response(response)
        stopwatch.print_elapsed_time()

    @exception_handling_void
    def unload_data(self, alloc_id, dataset_id):
        request = {
            "datasetId": dataset_id,
            "allocId": alloc_id
        }

        stopwatch = StopWatch()
        response = requests.post(f"{self.conf.gateway_base_url}/float/dataset/_unload",
                                 json=request,
                                 timeout=LONG_REQUEST_TIMEOUT_SECS)
        check_response(response)
        stopwatch.print_elapsed_time()
            
    @exception_handling
    def add_data(self, dataset_id, additional_data):
        request = {
            "datasetId": dataset_id,
            "recordsPath": additional_data
        }

        stopwatch = StopWatch()
        response = requests.post(f"{self.conf.gateway_base_url}/float/dataset/records/_add",
                                 json=request,
                                 timeout=LONG_REQUEST_TIMEOUT_SECS)

        check_response(response)

        stopwatch.print_elapsed_time()
        return response.json()["index"][0]
        
    def Search(self):
        return Float_Search(gsi=self)
            
    @exception_handling_void
    def remove_data(self, dataset_id, indices):
        if not isinstance(indices, list):
            indices = [indices]
        request = {
            "datasetId": dataset_id,
            "indices": indices
        }

        stopwatch = StopWatch()
        response = requests.post(f"{self.conf.gateway_base_url}/float/dataset/records/_remove",
                                   json=request,
                                   timeout=LONG_REQUEST_TIMEOUT_SECS)
        check_response(response)
        stopwatch.print_elapsed_time()
        
class Float_Search(object):        
    def __init__(self, gsi): 
        self.conf = gsi.conf
    
    def knn_composite_cosine(self, alloc_id, dataset_id, top_k, queries_path):
        try:
            validate_top_k(top_k)
        
            request = {
                "allocId": alloc_id,
                "datasetId": dataset_id,
                "topK": top_k,
                "queries": {
                    "filePath": queries_path
                }
            }

            stopwatch = StopWatch("knn_composite_cosine")
            response = requests.post(f"{self.conf.gateway_base_url}/float/_search",
                                     json=request,
                                     timeout=LONG_REQUEST_TIMEOUT_SECS)
            check_response(response)
            stopwatch.print_elapsed_time()
            index = np.array(response.json()["index"], dtype=np.int32)
            distance = np.array(response.json()["distance"], dtype=np.float32)
            return distance, index, response.json()["time"], gsi_api.get_ok_message()
        except Exception as e:
            logging.exception(e)
            return None, None, None, gsi_api.get_error_message()
