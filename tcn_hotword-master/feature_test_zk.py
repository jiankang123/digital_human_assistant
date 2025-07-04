import sys
import numpy as np

from feat_cpp.feature import FilterbankExtractor

if __name__ == '__main__':
    wav_path = sys.argv[1]
    filterbank_extractor = FilterbankExtractor()
    feature = filterbank_extractor.extract(wav_path)
    print(feature.shape)
    print(feature)