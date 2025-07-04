import torch
import numpy as np
from numpy import random
import torchaudio
import torchaudio.transforms as T


def specaug_torch(fbank, param):
    time_stretch_prob = param["time_stretch"]["prob"]
    time_masking_prob = param["time_masking"]["prob"]
    freq_masking_prob = param["freq_masking"]["prob"]
    
    if random.random() <= time_stretch_prob:
        stretch = T.TimeStretch()
        time_stretch_rate_low = param["time_stretch"]["rate_low"]
        time_stretch_rate_high = param["time_stretch"]["rate_high"]
        rate = random.randint(time_stretch_rate_low*10, time_stretch_rate_high*10) / 10
        fbank = stretch(fbank, rate)
    
    if random.random() <= time_masking_prob:
        time_mask_duration = param["time_masking"]["max_duration"]
        random_time_mask_duration = random.randint(0, time_mask_duration+1)
        masking = T.TimeMasking(time_mask_param=random_time_mask_duration)
        fbank = masking(fbank)

    if random.random() <= freq_masking_prob:
        freq_mask_duration = param["freq_masking"]["max_duration"]
        random_freq_mask_duration = random.randint(0, freq_mask_duration+1)
        masking = T.FrequencyMasking(freq_mask_param=random_freq_mask_duration)
        fbank = masking(fbank)
         
    return fbank

def test():
    fbank = torch.load("test_fbank.pt")
    speaug_param = {
        "time_stretch": {
            "prob": 0.5,
            "rate_low": 0.8,
            "rate_high": 1.2 
        },
        "time_masking": {
            "prob": 0.8,
            "max_duration": 20
        },
        "freq_masking": {
            "prob": 0.8,
            "max_duration": 3
        }
    }
    fbank_ = specaug_torch(fbank, speaug_param)

if __name__ == "__main__":
    test()

#TODO(fancui): only contains time & spec mask, need to add time warp
# def spec_augmentation(feature,
#                       max_time_mask=20,
#                       max_spec_mask=3,
#                       time_warp=False):

#     time_length = feature.shape[0]
#     spec_length = feature.shape[1]
#     random_time_duration = np.random.randint(0, max_time_mask+1)
#     random_time_start = np.random.randint(0, time_length - random_time_duration)
#     random_spec_duration = np.random.randint(0, max_spec_mask+1)
#     random_spec_start = np.random.randint(0, spec_length - random_spec_duration)
#     time_mask_probability = np.random.random()
#     spec_mask_probability = np.random.random()
    

#     random_time_duration = tf.cond(tf.greater(
#         time_mask_probability,
#         0.5), lambda: random_time_duration, lambda: tf.convert_to_tensor(0))
#     random_spec_duration = tf.cond(tf.greater(
#         spec_mask_probability,
#         0.5), lambda: random_spec_duration, lambda: tf.convert_to_tensor(0))
#     feature = tf.concat([
#         feature[:, :random_spec_start],
#         tf.zeros([time_length, random_spec_duration]),
#         feature[:, random_spec_start + random_spec_duration:]
#     ], 1)
#     feature = tf.concat([
#         feature[:random_time_start, :],
#         tf.zeros([random_time_duration, spec_length]),
#         feature[random_time_start + random_time_duration:, :]
#     ], 0)
#     return feature
