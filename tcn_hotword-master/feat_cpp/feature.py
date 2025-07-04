# Copyright 2019 Mobvoi Inc. All Rights Reserved.
# Author: shenli@mobvoi.com (Shen Li)

import numpy as np
import scipy.io.wavfile
import torch
try:
    from feat_cpp.filterbank_wrapper import Filterbank, FilterbankOptions, VectorFloat
except:
    from filterbank_wrapper import Filterbank, FilterbankOptions, VectorFloat


class FilterbankExtractor(object):
    def __init__(self):
        options = FilterbankOptions()
        self._filterbank = Filterbank(options)

    def extract(self, wav_path):
        _, audio = scipy.io.wavfile.read(wav_path)
        audio_len = len(audio)
        audio_as_vector = VectorFloat(audio_len)
        for i in range(audio_len):
            audio_as_vector[i] = audio.data[i]
        self._filterbank.Reset()
        return np.asarray(self._filterbank.Compute(audio_as_vector),
                          dtype=float)

def compute_cpp_filterbank_batch(audio):
    options = FilterbankOptions()
    filterbank = Filterbank(options)
    batch_size = audio.shape[0]
    audio_len = audio.shape[1]
    batch_feats = []
    for i in range(batch_size):
        audio_as_vector = VectorFloat(audio_len)
        for j in range(audio_len):
            audio_as_vector[j] = audio[i][j].item()
        filterbank.Reset()
        feat = np.asarray(filterbank.Compute(audio_as_vector), dtype=np.float32)
        batch_feats.append(feat)
    return torch.tensor(np.array(batch_feats))

if __name__ == "__main__":
    import sys
    wav_path = sys.argv[1]
    f = FilterbankExtractor()
    feature = f.extract(wav_path)
    print(feature.shape)
    print(feature)