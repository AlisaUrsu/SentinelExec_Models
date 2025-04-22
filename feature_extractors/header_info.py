import numpy as np
from sklearn.feature_extraction import FeatureHasher
from feature_extractors.feature_type import FeatureType


class HeaderFileInfo(FeatureType):
    ''' Machine, architecure, OS, linker and other information extracted from header 
        "header": {
            "coff": {
                "timestamp": timestamp in the header
                "machine": CPU architecture/string
                "characteristics": image characteristics/list of strings
            },
            "optional": {
                "subsystem": target subsystem/string
                "dll_characteristics": list of strings
                "magic": determines whether an image is a PE32 or PE32+ executable/string
                "major_image_version": int
                "minor_image_version": int
                "major_linker_version": int
                "minor_linker_version": int
                "major_operating_system_version": int
                "minor_operating_system_version": int
                "major_subsystem_version": int
                "minor_subsystem_version": int
                "sizeof_code": int
                "sizeof_headers": int
                "sizeof_heap_commit": int
            }
        }
    '''

    name = 'header'
    dim = 62

    def __init__(self):
        super(FeatureType, self).__init__()

    def raw_features(self, bytez, lief_binary):
        raw_obj = {}
        raw_obj['coff'] = {'timestamp': 0, 'machine': "", 'characteristics': []}
        raw_obj['optional'] = {
            'subsystem': "",
            'dll_characteristics': [],
            'magic': "",
            'major_image_version': 0,
            'minor_image_version': 0,
            'major_linker_version': 0,
            'minor_linker_version': 0,
            'major_operating_system_version': 0,
            'minor_operating_system_version': 0,
            'major_subsystem_version': 0,
            'minor_subsystem_version': 0,
            'sizeof_code': 0,
            'sizeof_headers': 0,
            'sizeof_heap_commit': 0
        }
        if lief_binary is None:
            return raw_obj

        raw_obj['coff']['timestamp'] = lief_binary.header.time_date_stamps
        raw_obj['coff']['machine'] = str(lief_binary.header.machine).split('.')[-1] # .head.machine returns something like "MACHINE_TYPES.I386"
        raw_obj['coff']['characteristics'] = [str(c).split('.')[-1] for c in lief_binary.header.characteristics_list] # same here; "CHARACTHERISTICS.DLL or .SYSTEM"
        raw_obj['optional']['subsystem'] = str(lief_binary.optional_header.subsystem).split('.')[-1]
        raw_obj['optional']['dll_characteristics'] = [
            str(c).split('.')[-1] for c in lief_binary.optional_header.dll_characteristics_lists
        ]
        raw_obj['optional']['magic'] = str(lief_binary.optional_header.magic).split('.')[-1]
        raw_obj['optional']['major_image_version'] = lief_binary.optional_header.major_image_version
        raw_obj['optional']['minor_image_version'] = lief_binary.optional_header.minor_image_version
        raw_obj['optional']['major_linker_version'] = lief_binary.optional_header.major_linker_version
        raw_obj['optional']['minor_linker_version'] = lief_binary.optional_header.minor_linker_version
        raw_obj['optional']['major_operating_system_version'] = lief_binary.optional_header.major_operating_system_version
        raw_obj['optional']['minor_operating_system_version'] = lief_binary.optional_header.minor_operating_system_version
        raw_obj['optional']['major_subsystem_version'] = lief_binary.optional_header.major_subsystem_version
        raw_obj['optional']['minor_subsystem_version'] = lief_binary.optional_header.minor_subsystem_version
        raw_obj['optional']['sizeof_code'] = lief_binary.optional_header.sizeof_code
        raw_obj['optional']['sizeof_headers'] = lief_binary.optional_header.sizeof_headers
        raw_obj['optional']['sizeof_heap_commit'] = lief_binary.optional_header.sizeof_heap_commit
        return raw_obj

    def process_raw_features(self, raw_obj):
        return np.hstack([
            raw_obj['coff']['timestamp'],
            FeatureHasher(10, input_type="string").transform([[raw_obj['coff']['machine']]]).toarray()[0],
            FeatureHasher(10, input_type="string").transform([raw_obj['coff']['characteristics']]).toarray()[0],
            FeatureHasher(10, input_type="string").transform([[raw_obj['optional']['subsystem']]]).toarray()[0],
            FeatureHasher(10, input_type="string").transform([raw_obj['optional']['dll_characteristics']]).toarray()[0],
            FeatureHasher(10, input_type="string").transform([[raw_obj['optional']['magic']]]).toarray()[0],
            raw_obj['optional']['major_image_version'],
            raw_obj['optional']['minor_image_version'],
            raw_obj['optional']['major_linker_version'],
            raw_obj['optional']['minor_linker_version'],
            raw_obj['optional']['major_operating_system_version'],
            raw_obj['optional']['minor_operating_system_version'],
            raw_obj['optional']['major_subsystem_version'],
            raw_obj['optional']['minor_subsystem_version'],
            raw_obj['optional']['sizeof_code'],
            raw_obj['optional']['sizeof_headers'],
            raw_obj['optional']['sizeof_heap_commit'],
        ]).astype(np.float32)
