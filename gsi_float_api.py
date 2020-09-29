import os
import requests
import numpy as np
import logging
import io
import time
import json
import h5py


def exception_handling(func):
    def func_wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs), "OK"
        except Exception as e:
            if "status=500" in str(e):
                logging.exception(e)
                return None, "Internal Server Error"
            return None, str(e)

    return func_wrapper


def exception_handling_void(func):
    def func_wrapper(*args, **kwargs):
        try:
            func(*args, **kwargs)
            return "OK"
        except Exception as e:
            if "status=500" in str(e):
                logging.exception(e)
                return "Internal Server Error"
            return str(e)

    return func_wrapper


class GSIConfig(object):
    FLOAT32_SERVER_HOST = os.getenv("FLOAT32_SERVER_HOST", "localhost")
    FLOAT32_SERVER_PORT = os.getenv("FLOAT32_SERVER_PORT", "7760")

    def __init__(self, log_level="INFO", float32_host=None, float32_port=None):
        self.log_level = log_level

        if float32_host is None:
            float32_host = self.FLOAT32_SERVER_HOST
        if float32_port is None:
            float32_port = self.FLOAT32_SERVER_PORT
        self.float32_base_url = f"http://{float32_host}:{float32_port}"


class Float_GSI(object):

    def __init__(self, conf=GSIConfig()):
        self.conf = conf
        log_level = logging.getLevelName(conf.log_level)
        setup_logger(log_level)

    @exception_handling
    def request_allocation(self, num_boards):
        request = {
            "numOfBoards": num_boards
        }
        response = requests.post(f"{self.conf.float32_base_url}/allocate", json=request)
        check_response(response)
        return response.json()["allocationId"]

    @exception_handling_void
    def delete_allocation(self, alloc_id):
        request = {
            "allocationId": alloc_id
        }
        response = requests.post(f"{self.conf.float32_base_url}/deallocate", json=request)
        check_response(response)

    @exception_handling
    def import_dataset(self, dataset_path, nbits=768):
        request = {
            "dsFilePath": dataset_path,
            "nbits": nbits
        }
        response = requests.post(f"{self.conf.float32_base_url}/import/dataset", json=request)
        check_response(response)
        return response.json()["datasetId"]

    @exception_handling_void
    def load_data(self, alloc_id, dataset_id, typical_nqueries=1, max_nqueries=1):
        request = {
            "allocationId": alloc_id,
            "datasetId": dataset_id,
            "typicalNQueries": typical_nqueries,
            "maxNQueries": max_nqueries
        }
        response = requests.post(f"{self.conf.float32_base_url}/loadDataset", json=request)
        check_response(response)

    @exception_handling_void
    def unload_data(self, alloc_id, dataset_id):
        request = {
            "allocationId": alloc_id,
            "datasetId": dataset_id,
        }
        response = requests.post(f"{self.conf.float32_base_url}/unloadDataset", json=request)
        check_response(response)

    @exception_handling
    def add_data(self, dataset_id, additional_data):
        request = {
            "datasetId": dataset_id,
            "dataToAdd": additional_data
        }
        response = requests.post(f"{self.conf.float32_base_url}/addData", json=request)
        check_response(response)
        return response.json()["datasetPath"]

    @exception_handling
    def remove_data(self, dataset_id, indices):
        request = {
            "datasetId": dataset_id,
            "indicesToRemove": indices
        }
        response = requests.post(f"{self.conf.float32_base_url}/removeData", json=request)
        check_response(response)
        return response.json()["datasetPath"]

    def Search(self):
        return Float_Search(conf=self.conf)


class Float_Search(object):

    def __init__(self, conf):
        self.conf = conf

    def knn_composite_cosine(self, alloc_id, dataset_id, queries_path, topk):
        request = {
            "allocationId": alloc_id,
            "datasetId": dataset_id,
            "queriesPath": queries_path,
            "topk": topk
        }

        response = requests.post(f"{self.conf.float32_base_url}/search", json=request)

        try:
            check_response(response)
            index = np.array(response.json()["indices"], dtype=np.long)
            distance = np.array(response.json()["distance"], dtype=np.float32)
            return distance, index, response.json()["search"], "OK"
        except Exception as e:
            if "status=500" in str(e):
                logging.exception(e)
                return None, None, None, "Internal Server Error"
            return None, None, None, str(e)

    def knn_composite_cosine_by_queries(self, alloc_id, dataset_id, queries, topk):
        request = {
            "allocationId": alloc_id,
            "datasetId": dataset_id,
            "queries": queries,
            "topk": topk
        }

        response = requests.post(f"{self.conf.float32_base_url}/searchByQueriesList", json=request)

        try:
            check_response(response)
            index = np.array(response.json()["indices"], dtype=np.long)
            distance = np.array(response.json()["distance"], dtype=np.float32)
            return distance, index, response.json()["search"], "OK"
        except Exception as e:
            if "status=500" in str(e):
                logging.exception(e)
                return None, None, None, "Internal Server Error"
            return None, None, None, str(e)

    def knn_composite_cosine_by_hdf(self, alloc_id, dataset_id, hdf_queries_path, topk):
        request = {
            "allocationId": alloc_id,
            "datasetId": dataset_id,
            "queriesPath": hdf_queries_path,
            "topk": topk
        }

        response = requests.post(f"{self.conf.float32_base_url}/searchByHdf", json=request)

        try:
            check_response(response)
            index = np.array(response.json()["indices"], dtype=np.long)
            distance = np.array(response.json()["distance"], dtype=np.float32)
            return distance, index, response.json()["search"], "OK"
        except Exception as e:
            if "status=500" in str(e):
                logging.exception(e)
                return None, None, None, "Internal Server Error"
            return None, None, None, str(e)


def check_response(response):
    if response.status_code == 400:
        raise Exception(response.content.decode('utf-8'))
    if response.status_code != requests.codes.ok:  # pylint: disable=E1101
        raise Exception(f"Error occurred during operation. status={response.status_code}.")


def setup_logger(log_level):
    logging.basicConfig(level=log_level)
    ch = logging.StreamHandler()
    ch.setFormatter(CustomLoggingFormatter())
    logging.getLogger().addHandler(ch)


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
    msg_format = "%(levelname)s:%(asctime)s:%(name)s - %(message)s"

    FORMATS = {
        logging.DEBUG: white + msg_format + reset,
        logging.INFO: info + msg_format + reset,
        logging.WARNING: bright_red + msg_format + reset,
        logging.ERROR: bright_red + msg_format + reset,
        logging.CRITICAL: red + msg_format + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)
