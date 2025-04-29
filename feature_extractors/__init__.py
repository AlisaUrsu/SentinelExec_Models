import multiprocessing
import os
import tqdm 
import numpy as np
import pandas as pd
import json
from feature_extractors.pe_feature import PEFeatureExtractor


def raw_feature_iterator(file_paths):
    """
    Yield raw feature strings from the inputed file paths
    """
    for path in file_paths:
        with open(path, "r") as fin:
            for line in fin:
                try:
                    obj = json.loads(line)
                    if obj.get("label") in (0, 1):
                        yield line
                except Exception as e:
                    print(f"Skipping invalid line: {e}")

def count_filtered_lines(file_paths):
    count = 0
    for _ in raw_feature_iterator(file_paths):
        count += 1
    return count

def write_filtered_features(raw_feature_paths, output_path):
    with open(output_path, "w") as fout:
        for fp in raw_feature_paths:
            with open(fp, "r") as fin:
                for line in fin:
                    try:
                        obj = json.loads(line)
                        if obj.get("label") in (0, 1):
                            fout.write(line)
                    except Exception as e:
                        print(f"Skipping invalid line: {e}")


def vectorize(irow, raw_features_string, X_path, y_path, extractor, nrows):
    """
    Vectorize a single sample of raw features and write to a large numpy file
    """
    raw_features = json.loads(raw_features_string)
    feature_vector = extractor.process_raw_features(raw_features)

    y = np.memmap(y_path, dtype=np.float32, mode="r+", shape=nrows)
    y[irow] = raw_features["label"]

    X = np.memmap(X_path, dtype=np.float32, mode="r+", shape=(nrows, extractor.dim))
    X[irow] = feature_vector


def vectorize_unpack(args):
    """
    Pass through function for unpacking vectorize arguments
    """
    return vectorize(*args)

def vectorize_subset(X_path, y_path, raw_feature_paths, extractor, nrows):
    """
    Vectorize a subset of data and write it to disk
    """
    X = np.memmap(X_path, dtype=np.float32, mode="w+", shape=(nrows, extractor.dim))
    y = np.memmap(y_path, dtype=np.float32, mode="w+", shape=nrows)
    del X, y

    argument_iterator = ((irow, raw_features_string, X_path, y_path, extractor, nrows)
                         for irow, raw_features_string in enumerate(raw_feature_iterator(raw_feature_paths)))
    for args in tqdm.tqdm(argument_iterator, total=nrows):
        vectorize_unpack(args)

def create_vectorized_features(data_dir):
    """
    Create feature vectors from raw features and write them to disk
    """
    extractor = PEFeatureExtractor()
    raw_feature_paths = [os.path.join(data_dir, f"train_features_{i}.jsonl") for i in range(6)]
    filtered_path = os.path.join(data_dir, "filtered_train.jsonl")

    write_filtered_features(raw_feature_paths, filtered_path)


    print("Vectorizing training set")
    X_path = os.path.join(data_dir, "X_train.dat")
    y_path = os.path.join(data_dir, "y_train.dat")
    nrows = sum(1 for _ in open(filtered_path)) 
    vectorize_subset(X_path, y_path, raw_feature_paths, extractor, nrows)

    print("Vectorizing testing set")
    X_path = os.path.join(data_dir, "X_test.dat")
    y_path = os.path.join(data_dir, "y_test.dat")
    raw_feature_paths = [os.path.join(data_dir, "test_features.jsonl")]
    nrows = sum([1 for fp in raw_feature_paths for line in open(fp)])
    vectorize_subset(X_path, y_path, raw_feature_paths, extractor, nrows)


def read_vectorized_features(data_dir, subset=None):
    """
    Read vectorized features into memory mapped numpy arrays
    """
    extractor = PEFeatureExtractor()
    ndim = extractor.dim
    X_train = None
    y_train = None
    X_test = None
    y_test = None

    X_train_path = os.path.join(data_dir, "X_train.dat")
    y_train_path = os.path.join(data_dir, "y_train.dat")
    y_train = np.memmap(y_train_path, dtype=np.float32, mode="r")
    N = y_train.shape[0]
    X_train = np.memmap(X_train_path, dtype=np.float32, mode="r", shape=(N, ndim))

    X_test_path = os.path.join(data_dir, "X_test.dat")
    y_test_path = os.path.join(data_dir, "y_test.dat")
    y_test = np.memmap(y_test_path, dtype=np.float32, mode="r")
    N = y_test.shape[0]
    X_test = np.memmap(X_test_path, dtype=np.float32, mode="r", shape=(N, ndim))

    return X_train, y_train, X_test, y_test
