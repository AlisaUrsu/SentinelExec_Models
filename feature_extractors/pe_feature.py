import hashlib
import json
import os

import lief
import numpy as np
import pandas as pd
import json
from feature_extractors.byte_entropy_histogram import ByteEntropyHistogram
from feature_extractors.byte_histogram import ByteHistogram
from feature_extractors.exports import ExportsInfo
from feature_extractors.general_info import GeneralFileInfo
from feature_extractors.header_info import HeaderFileInfo
from feature_extractors.imports import ImportsInfo
from feature_extractors.sections import SectionInfo
from feature_extractors.strings import StringExtractor


class PEFeatureExtractor(object):
    ''' Extract useful features from a PE file, and return as a vector of fixed size. '''

    def __init__(self, features_file=''):
        self.features = []
        features = {
                    'histogram': ByteHistogram(),
                    'byteentropy': ByteEntropyHistogram(),
                    'strings': StringExtractor(),
                    'general': GeneralFileInfo(),
                    'header': HeaderFileInfo(),
                    'section': SectionInfo(),
                    'imports': ImportsInfo(),
                    'exports': ExportsInfo()
            }

        if os.path.exists(features_file):

            '''with open(features_file, encoding='utf8') as f:
                df = pd.read_parquet(features_file)
                # Assuming the .parquet file has a column "features" with a list in the first row
                feature_list = df.iloc[0]['features']
                self.features = [features[feature] for feature in feature_list if feature in features]'''
            with open(features_file, encoding='utf8') as f:
                x = json.load(f)
                self.features = [features[feature] for feature in x['features'] if feature in features]
        else:
            self.features = list(features.values())

        self.dim = sum([fe.dim for fe in self.features])

    def raw_features(self, bytez):
        lief_errors = (
                       RuntimeError)
        try:
            lief_binary = lief.PE.parse(list(bytez))
        except lief_errors as e:
            print("lief error: ", str(e))
            lief_binary = None
        except Exception:  # everything else (KeyboardInterrupt, SystemExit, ValueError):
            raise

        features = {"sha256": hashlib.sha256(bytez).hexdigest()}
        features.update({fe.name: fe.raw_features(bytez, lief_binary) for fe in self.features})
        return features

    def process_raw_features(self, raw_obj):
        feature_vectors = [fe.process_raw_features(raw_obj[fe.name]) for fe in self.features]
        return np.hstack(feature_vectors).astype(np.float32)

    def feature_vector(self, bytez):
        return self.process_raw_features(self.raw_features(bytez))