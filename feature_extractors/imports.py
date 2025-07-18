import numpy as np
from sklearn.feature_extraction import FeatureHasher

from feature_extractors.feature_type import FeatureType


class ImportsInfo(FeatureType):
    ''' Information about imported libraries and functions from the
    import address table.  Note that the total number of imported
    functions is contained in GeneralFileInfo.
    '''

    name = 'imports'
    dim = 1280

    def __init__(self):
        super(FeatureType, self).__init__()

    def raw_features(self, bytez, lief_binary):
        imports = {}
        if lief_binary is None:
            return imports
        for lib in lief_binary.imports:
            if lib.name not in imports:
                imports[lib.name] = []  # libraries can be duplicated in listing, extend instead of overwrite

            # Clipping assumes there are diminishing returns on the discriminatory power of imported functions
            #  beyond the first 10000 characters, and this will help limit the dataset size
            for entry in lib.entries:
                if entry.is_ordinal:
                    imports[lib.name].append("ordinal" + str(entry.ordinal))
                else:
                    imports[lib.name].append(entry.name[:10000])

        return imports



    def process_raw_features(self, raw_obj):
        for lib, funcs in raw_obj.items():
            for f in funcs:
                if f is None:
                    print("Found None in import function list!")
                elif not isinstance(f, str):
                    print("Non-string import:", type(f), f)
        # unique libraries
        libraries = list(set([l.lower() for l in raw_obj.keys()]))
        libraries_hashed = FeatureHasher(256, input_type="string").transform([libraries]).toarray()[0]

        # A string like "kernel32.dll:CreateFileMappingA" for each imported function
        imports = [lib.lower() + ':' + e for lib, elist in raw_obj.items() for e in elist]
        imports_hashed = FeatureHasher(1024, input_type="string").transform([imports]).toarray()[0]

        # Two separate elements: libraries (alone) and fully-qualified names of imported functions
        return np.hstack([libraries_hashed, imports_hashed]).astype(np.float32)