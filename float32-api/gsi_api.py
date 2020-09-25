import os
from os.path import isfile, isdir, join
from pathlib import Path
import time
import requests
from datetime import datetime
import json
from os import path
import numpy as np
import inspect
import message_codes as messages
from message_codes import Messages
import base64
import socket
import logging

DEFAULT_GATEWAY_SERVER_HOST = os.getenv("GATEWAY_SERVER_HOST", "localhost")
DEFAULT_GATEWAY_SERVER_PORT = os.getenv("GATEWAY_SERVER_PORT", "8099")
SHORT_REQUEST_TIMEOUT_SECS = int(os.getenv("SHORT_REQUEST_TIMEOUT_SECS", "5"))
LONG_REQUEST_TIMEOUT_SECS = int(os.getenv("LONG_REQUEST_TIMEOUT_SECS", "3600"))

DEFAULT_SEARCH_API_TCP_HOST = os.getenv("SEARCH_API_TCP_HOST", "localhost")
DEFAULT_SEARCH_API_TCP_PORT = int(os.getenv("SEARCH_API_TCP_PORT", "8110"))
SEARCH_API_TCP_READ_CHUNK_SIZE = 1000000

def setup_logger(log_level):
    logging.basicConfig(level=log_level)
    
    ch = logging.StreamHandler()
    ch.setFormatter(CustomLoggingFormatter())
    logging.getLogger().addHandler(ch)
    
def exception_handling(func):
    def func_wrapper(*args, **kwargs):
        try:
           return func(*args, **kwargs), get_ok_message()

        except Exception as e:
            logging.exception(e)
            return None, get_error_message(str(e))

    return func_wrapper
    
def exception_handling_void(func):
    def func_wrapper(*args, **kwargs):
        try:
            func(*args, **kwargs)
            return get_ok_message()

        except Exception as e:
            logging.exception(e)
            return get_error_message(str(e))

    return func_wrapper

def check_response(response):
    if response.status_code == 400:
        raise Exception(response.content.decode('utf-8'))
    if response.status_code != requests.codes.ok:
        raise Exception(f"Error occurred during operation. status={response.status_code}.")

def get_ok_message():
    return list(Messages.get_dict(messages.OK).values())[messages.STR_INDEX]

def get_error_message(message=None):
    if message is None or 'status=500' in message:
        return list(Messages.get_dict(messages.GENERAL_ERROR).values())[messages.STR_INDEX]
    return message

def validate_num_boards(num_boards):
    if type(num_boards) != int:
        raise Exception("num_boards must be int")
        
def validate_data_type(data_type):
        if data_type != np.int16:
            raise Exception(f"data_type {str(data_type)} is not supported")
            
def get_num_of_features(nbits, data_type):
    return int(nbits / (np.dtype(data_type).itemsize * 8))
        
class GSIConfig(object):
    def __init__(self, log_level="INFO", sm_host=None, gateway_host=None, gateway_port=None, search_host=None, search_port=None):
        self.log_level = log_level
        
        if gateway_host is None:
            gateway_host = sm_host
        if search_host is None:
            search_host = sm_host
            
        gateway_host = self.get_parameter_or_env_variable(gateway_host, DEFAULT_GATEWAY_SERVER_HOST)
        gateway_port = self.get_parameter_or_env_variable(gateway_port, DEFAULT_GATEWAY_SERVER_PORT)
        self.gateway_base_url = f"http://{gateway_host}:{gateway_port}"
        
        self.search_api_tcp_host = self.get_parameter_or_env_variable(search_host, DEFAULT_SEARCH_API_TCP_HOST)
        self.search_api_tcp_port = self.get_parameter_or_env_variable(search_port, DEFAULT_SEARCH_API_TCP_PORT)
        
    @classmethod
    def get_parameter_or_env_variable(cls, param, env_variable):
        if (param is not None):
            return param
        return env_variable
        

class GSI(object):
    def __init__(self, conf=GSIConfig()):
        self.conf = conf
        log_level = logging.getLevelName(conf.log_level)
        setup_logger(log_level)

    @exception_handling
    def request_allocation(self, num_boards, name=None):
        validate_num_boards(num_boards)
        request = {
            "name":name,
            "numOfBoards": num_boards
        }
        stopwatch = StopWatch(name="request_allocation")
        response = requests.post(f"{self.conf.gateway_base_url}/resource/allocation/_request",
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
        response = requests.delete(f"{self.conf.gateway_base_url}/resource/allocation/{alloc_id}",
                                 timeout=SHORT_REQUEST_TIMEOUT_SECS)
        check_response(response)
        stopwatch.print_elapsed_time()

    @exception_handling
    def allocate_resources(self, name, boards):
        request = {
            "allocName": name,
            "boardAllocations": boards
        }

        stopwatch = StopWatch(name="allocate_resources")
        response = requests.put(f"{self.conf.gateway_base_url}/resource/allocation",
                                 json=request,
                                 timeout=SHORT_REQUEST_TIMEOUT_SECS)
        check_response(response)
        stopwatch.print_elapsed_time()
        return response.json()["allocId"]

    @exception_handling
    def import_dataset(self, data_type=np.int16, nbits=None, recordsFile=None, indicesFile=None, name=None):
        validate_data_type(data_type)
    
        basename = os.path.basename(recordsFile)
        ext = ''.join(basename.split('.')[1:])
        ext = ext if ext else "bin"
        
        num_features = get_num_of_features(nbits, data_type)
    
        request = {
            "type": "DATASET",
            "recordsType": np.dtype(data_type).name,
            "recordsFormat": ext,
            "name": name,
            "indicesType": "int32",
            "queryLength": num_features,
            "params": {
                "parts": [
                    {
                        "recordsFile":recordsFile,
                        "indicesFile":indicesFile
                    }
                ]
            }
        }

        stopwatch = StopWatch(name="import_dataset")
        response = requests.post(f"{self.conf.gateway_base_url}/dataset/_import",
                                 json=request,
                                 timeout=SHORT_REQUEST_TIMEOUT_SECS)
        check_response(response)
        stopwatch.print_elapsed_time()
        return response.json()["id"]

    @exception_handling_void
    def load_data(self, alloc_id, dataset_id):
        request = {
            "datasetId": dataset_id,
            "allocId": alloc_id
        }
        print(f"load_data req: {str(request)}")

        stopwatch = StopWatch(name="load_data")
        response = requests.post(f"{self.conf.gateway_base_url}/dataset/_load",
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

        stopwatch = StopWatch(name="unload_data")
        response = requests.post(f"{self.conf.gateway_base_url}/dataset/_unload",
                                 json=request,
                                 timeout=LONG_REQUEST_TIMEOUT_SECS)
        check_response(response)
        stopwatch.print_elapsed_time()
        
    @exception_handling_void
    def unload_all_data(self):
        stopwatch = StopWatch(name="unload_data")
        response = requests.post(f"{self.conf.gateway_base_url}/dataset/_unloadAll",
                                 timeout=LONG_REQUEST_TIMEOUT_SECS)
        check_response(response)
        stopwatch.print_elapsed_time()
        
    @exception_handling
    def get_dataset(self, dataset_id):
        response = requests.get(f"{self.conf.gateway_base_url}/dataset/" + dataset_id,
                                timeout=SHORT_REQUEST_TIMEOUT_SECS)
        check_response(response)
        return response.json()
        
    @exception_handling
    def get_datasets(self):
        response = requests.get(f"{self.conf.gateway_base_url}/dataset",
                                 timeout=LONG_REQUEST_TIMEOUT_SECS)
        check_response(response)
        return response.json()
        
    @exception_handling
    def get_allocations(self):
        response = requests.get(f"{self.conf.gateway_base_url}/resource/allocation",
                                 timeout=LONG_REQUEST_TIMEOUT_SECS)
                                 
        check_response(response)
        return response.json()["allocations"]
        
    def Search(self):
        return Search(gsi=self)

class Search(object):
    def open_socket(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        logging.info(f"Connecting tcp socket to {self.conf.search_api_tcp_host}:{self.conf.search_api_tcp_port} ...")
        self.sock.connect((self.conf.search_api_tcp_host, self.conf.search_api_tcp_port))
        
    def close_socket(self):
        self.sock.close()
        logging.info(f"Closed tcp socket to {self.conf.search_api_tcp_host}:{self.conf.search_api_tcp_port}")

    def __init__(self, gsi): 
        self.conf = gsi.conf
        self.open_socket()
        
    def __del__(self):
        self.close_socket()
        
    def core_search(self, alloc_id, dataset_id, top_k, algorithm, query):    
        request = {
            "allocId": alloc_id,
            "datasetId": dataset_id,
            "query": {
                "queryAsBase64": query.queryAsBase64,
                "features": query.features,
                "recordType":"int16"
            },
            "algoritm": algorithm,
            "topK": top_k
        }
        
        send_obj = {
            "key": "search",
            "request": request
        }
        request_str = json.dumps(send_obj) + "\n"
        
        stopwatch = StopWatch(name=f"search {algorithm}")
        self.sock.send(request_str.encode("utf-8"))
        
        first_exec = True
        total_data = b''
        while ((first_exec and total_data == b'') or ((data is not None and len(data) > 0) and data[-1] != 10)):
            first_exec = False
            data = self.sock.recv(SEARCH_API_TCP_READ_CHUNK_SIZE)
            total_data += data
        response_str = total_data.decode("utf-8")
        raw_response = json.loads(response_str)
        
        scores = [
            [record["score"] for record in query_result["results"]]
            for query_result in raw_response["queryResults"]
        ]
        indices = [
            [record["index"] for record in query_result["results"]]
            for query_result in raw_response["queryResults"]
        ]
        
        return scores, indices, get_ok_message()

    def knn_tanimoto(self, alloc_id, dataset_id, top_k, query):
        try:
            return self.core_search(alloc_id, dataset_id, top_k ,"tanimoto", query)
        except Exception as e:
            logging.exception(e)
            return None,None, get_error_message(str(e))

    def knn_hamming(self, alloc_id, dataset_id, top_k, query):
        try:
            return self.core_search(alloc_id, dataset_id, top_k ,"hamming", query)
        except Exception as e:
            logging.exception(e)
            return None,None, get_error_message(str(e))
        
class Query(object):
    def __init__(self, queryAsBase64, nbits, data_type):
        self.queryAsBase64 = queryAsBase64
        self.features = get_num_of_features(nbits, data_type)
        
    @classmethod
    def bin_file_to_base64(cls, path):
        with open(path, "rb") as f:
            file_as_base64 = base64.b64encode(f.read())
            return file_as_base64.decode()
       
    @classmethod
    def np_arr_to_base64(cls, np_arr):
        return base64.b64encode(np_arr).decode("utf-8")
    
    @classmethod
    def from_file(cls, file_path, nbits, data_type=np.int16):
        queryAsBase64 = cls.bin_file_to_base64(file_path)
        return cls(queryAsBase64, nbits, data_type)
            
    @classmethod
    def from_numpy_array(cls, query_arr):
        queryAsBase64 = cls.np_arr_to_base64(query_arr)
        record_length = query_arr.shape[1]
        data_type = query_arr.dtype
        nbits = record_length * np.dtype(data_type).itemsize * 8
        
        return cls(queryAsBase64, nbits, data_type)

class StopWatch:
    def __init__(self, name=inspect.currentframe().f_code.co_name, start=True):
        self.start_time = 0
        self.stop_time = 0
        self.op_name = name
        if start:
            self.start()

    def start(self):
        self.start_time = time.time()
        return self.start_time

    def stop(self):
        self.stop_time = time.time()
        return self.stop_time

    def get_elapsed_time(self):
        self.stop()
        return (self.stop_time - self.start_time)

    def print_elapsed_time(self):
        logging.debug(f"{self.op_name} took {format(self.get_elapsed_time(), '.6f')} seconds")

class CustomLoggingFormatter(logging.Formatter):
    bblackforg = "\x1b[100m"
    bredforg = "\x1b[101m"
    info = "\x1b[33m"
    brightyellow = "\x1b[33m"
    yellow = "\x1b[33m"
    bright_yellow = "\x1b[93m"

    red = "\x1b[31m"
    bright_red = "\x1b[91m"

    black = "\x1b[90m"
    bold_red = "\x1b[31m"
    white = "\x1b[0m"
    reset = "\033[0m"
    format = "%(levelname)s:%(asctime)s:%(name)s - %(message)s"

    FORMATS = {
        logging.DEBUG: white + format + reset,
        logging.INFO: info + format + reset,
        logging.WARNING: bright_red + format + reset,
        logging.ERROR: bright_red + format + reset,
        logging.CRITICAL: red + format + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)