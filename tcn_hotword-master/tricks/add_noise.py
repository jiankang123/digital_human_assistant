import torch
from torch.utils.data import DataLoader
import os
import numpy as np
from scipy.io import wavfile


class AddNoise(torch.nn.Module):
    """This class additively combines a noise signal to the input signal.
    Arguments
    ---------
    csv_file : str
        The name of a csv file containing the location of the
        noise audio files. If none is provided, white noise will be used.
    csv_keys : list, None, optional
        Default: None . One data entry for the noise data should be specified.
        If None, the csv file is expected to have only one data entry.
    sorting : str
        The order to iterate the csv file, from one of the
        following options: random, original, ascending, and descending.
    num_workers : int
        Number of workers in the DataLoader (See PyTorch DataLoader docs).
    snr_low : int
        The low end of the mixing ratios, in decibels.
    snr_high : int
        The high end of the mixing ratios, in decibels.
    pad_noise : bool
        If True, copy noise signals that are shorter than
        their corresponding clean signals so as to cover the whole clean
        signal. Otherwise, leave the noise un-padded.
    mix_prob : float
        The probability that a batch of signals will be mixed
        with a noise signal. By default, every batch is mixed with noise.
    start_index : int
        The index in the noise waveforms to start from. By default, chooses
        a random index in [0, len(noise) - len(waveforms)].
    normalize : bool
        If True, output noisy signals that exceed [-1,1] will be
        normalized to [-1,1].
    replacements : dict
        A set of string replacements to carry out in the
        csv file. Each time a key is found in the text, it will be replaced
        with the corresponding value.
    noise_sample_rate : int
        The sample rate of the noise audio signals, so noise can be resampled
        to the clean sample rate if necessary.
    clean_sample_rate : int
        The sample rate of the clean audio signals, so noise can be resampled
        to the clean sample rate if necessary.
    Example
    -------
    >>> import pytest
    >>> from speechbrain.dataio.dataio import read_audio
    >>> signal = read_audio('tests/samples/single-mic/example1.wav')
    >>> clean = signal.unsqueeze(0) # [batch, time, channels]
    >>> noisifier = AddNoise('tests/samples/annotation/noise.csv',
    ...                     replacements={'noise_folder': 'tests/samples/noise'})
    >>> noisy = noisifier(clean, torch.ones(1))
    """

    def __init__(
        self,
        noise_dir,
        num_workers=0,
        snr_low=0,
        snr_high=0,
        pad_noise=False,
        mix_prob=1.0,
        start_index=None,
        normalize=False,
        replacements={},
        noise_sample_rate=16000,
        clean_sample_rate=16000,
        noise_type=None,
        batch_size=None
    ):
        super().__init__()

        self.num_workers = num_workers
        self.snr_low = snr_low
        self.snr_high = snr_high
        self.pad_noise = pad_noise
        self.mix_prob = mix_prob
        self.start_index = start_index
        self.normalize = normalize
        self.replacements = replacements
        self.noise_type = noise_type
        self.noise_dir = noise_dir
        self.batch_size = batch_size

        if noise_sample_rate != clean_sample_rate:
            self.resampler = Resample(noise_sample_rate, clean_sample_rate)

    def forward(self, waveforms, lengths):
        """
        Arguments
        ---------
        waveforms : tensor
            Shape should be `[batch, time]` or `[batch, time, channels]`.
        lengths : tensor
            Shape should be a single dimension, `[batch]`.
        Returns
        -------
        Tensor of shape `[batch, time]` or `[batch, time, channels]`.
        """
        self.batch_size = len(lengths)
        # waveforms = waveforms*1.2

        # Copy clean waveform to initialize noisy waveform
        noisy_waveform = waveforms.clone().float()
        # Don't add noise (return early) 1-`mix_prob` portion of the batches
        if torch.rand(1) > self.mix_prob:
            return noisy_waveform

        # Compute the average amplitude of the clean waveforms
        clean_power = compute_amplitude(waveforms,lengths,amp_type="avg")
        # print('clean compute amplitude done')
        # print(clean_power.shape)
        # Pick an SNR and use it to compute the mixture amplitude factors
    

        SNR = torch.randint(low=self.snr_low, high=self.snr_high, size=(self.batch_size,), device=waveforms.device)
        # SNR=torch.tensor([0,0,0,0,0])
        factor = 1/dB_to_amplitude(SNR)
        # print('factor',factor)

        # Loop through clean samples and create mixture
        if True:
            tensor_length = waveforms.shape[1]
            noise_waveform, noise_length, noise_names = self._load_noise(
                lengths, tensor_length,
            )
            # print('noise load done')
            # for i,d in enumerate(noise_waveform):
                # wavfile.write('/ssd1/qiang/test_aug/{}.wav'.format(i), 16000, d.numpy().astype(np.int16))
            # Rescale and add
            # print(noise_names)
            noise_power = compute_amplitude(noise_waveform, noise_length,amp_type="avg")
            # print('noise compute amplitude done')
            # print(noise_power.shape)
            # print(clean_power)
            # print(factor.unsqueeze(1))
            # print(noise_power.unsqueeze(1))
            # x = (clean / noise) * (1/[DB2AMP(SNR)-1])

            noise_factor=clean_power.unsqueeze(1)*factor.unsqueeze(1)/noise_power.unsqueeze(1)
            # print(noise_factor.shape)
            # clean + x * noise
            # print(noise_factor)
            new_noise = noise_factor*noise_waveform
            # print(new_noise.shape)
            # print(new_noise)
            # for no, name, l in zip(new_noise,noise_names,lengths):
                # wavfile.write('/ssd1/qiang/test_aug/{}'.format(name), 16000, no[:l].numpy().astype(np.int16))
            noisy_waveform += new_noise

        # Normalizing to prevent clipping
        if self.normalize:
            abs_max, _ = torch.max(
                torch.abs(noisy_waveform), dim=1, keepdim=True
            )
            noisy_waveform = noisy_waveform / abs_max.clamp(min=1.0)

        return noisy_waveform, noise_names, SNR

    def _load_noise(self, lengths, max_length):
        """Load a batch of noises"""
        # batch_size = 1

        # Load a noise batch
        if not hasattr(self, "data_loader"):
            # Set parameters based on input
            self.device = lengths.device

            # Create a data loader for the noise wavforms
            dataset = my_noise_dataset(self.noise_dir)
            
            self.data_loader = DataLoader(
                    dataset, 
                    batch_size=self.batch_size, 
                    shuffle=True, 
                    sampler=None, 
                    num_workers=self.num_workers, 
                    collate_fn=self.Collate_fn, 
                    pin_memory=False, 
                    drop_last=False,
                )
            self.noise_data = iter(self.data_loader)
        # Load noise to correct device
        noise_batch, noise_len, noise_names = self._load_noise_batch_of_size(len(lengths))
        # noise_batch = torch.tensor([x.tolist() for x in noise_batch])
        # print(noise_batch)
        # for i,d in enumerate(noise_batch):
        #     wavfile.write('/ssd1/qiang/test_aug/{}.wav'.format(i), 16000, np.array(d).astype(np.int16))
        noise_batch = noise_batch.to(lengths.device)
        noise_len = noise_len.to(lengths.device)

        # Ensure shortest wav can cover speech signal
        # WARNING: THIS COULD BE SLOW IF THERE ARE VERY SHORT NOISES
        # print('pad noise')
        if self.pad_noise:
            while torch.any(noise_len < lengths):
                # min_len = torch.min(noise_len)
                prepend = noise_batch
                noise_batch = torch.cat((prepend, noise_batch), axis=1)
                noise_len += noise_len
        # Ensure noise batch is long enough
        elif noise_batch.size(1) < max_length:
            padding = (0, max_length - noise_batch.size(1))
            noise_batch = torch.nn.functional.pad(noise_batch, padding)
        # Select a random starting location in the waveform
        # for i,d in enumerate(noise_batch):
        #     wavfile.write('/ssd1/qiang/test_aug/{}.wav'.format(i), 16000, np.array(d).astype(np.int16))
        start_index = self.start_index
        if self.start_index is None:
            start_index = 0
            max_chop = (noise_len - lengths).min().clamp(min=1)
            start_index = torch.randint(low=0,
                high=int(max_chop.item()), size=(1,), device=lengths.device
            ).item()

        # Truncate noise_batch to max_length
        noise_batch = noise_batch[:, start_index : start_index + max_length]
        noise_len = (noise_len - start_index).clamp(max=max_length).unsqueeze(1)
        return noise_batch, noise_len.squeeze(1), noise_names

    def Collate_fn(self, batch_data):
        batch_wav = []
        for item in batch_data:
            sr, data = wavfile.read(item)
            batch_wav.append((data,item))
        batch_wav.sort(key=lambda x: len(x[0]),reverse=True)
        lengths = [len(x[0]) for x in batch_wav]
        maxlen = lengths[0]
        noise_name = [x[1].split('/')[-1] for x in batch_wav]
        batch_wav = [x[0] for x in batch_wav]
        data = [batch_wav[0]]
        for d in range(1,len(batch_wav)):
            if maxlen%lengths[d]==0:
                times = int(maxlen/lengths[d])
            else:
                times = int(maxlen/lengths[d])+1
            ori_data = batch_wav[d]
            ori_data = np.tile(ori_data,(1,times))[0]
            # wavfile.write('/ssd1/qiang/test_aug/{}.wav'.format(d), 16000, np.array(ori_data[:maxlen]).astype(np.int16))
            data.append(ori_data[:maxlen].astype(np.int16))
        data = np.array(data)
        data = torch.tensor(data)

        return data,torch.tensor(lengths),noise_name

    def _load_noise_batch_of_size(self, batch_size):
        """Concatenate noise batches, then chop to correct size"""
        # print('in _load_noise_batch_of_size')
        noise_batch, noise_lens, noise_names = self._load_noise_batch()
        # print('_load_noise_batch_of_size')

        # Expand
        while len(noise_batch) < batch_size:
            added_noise, added_lens, added_noise_names = self._load_noise_batch()
            noise_batch, noise_lens, noise_names = AddNoise._concat_batch(
                noise_batch, noise_lens, noise_names, added_noise, added_lens, added_noise_names
            )

        # Contract
        if len(noise_batch) > batch_size:
            noise_batch = noise_batch[:batch_size]
            noise_lens = noise_lens[:batch_size]
        return noise_batch, noise_lens, noise_names

    @staticmethod
    def _concat_batch(noise_batch, noise_lens, noise_names, added_noise, added_lens, added_noise_names):
        """Concatenate two noise batches of potentially different lengths"""

        # pad shorter batch to correct length
        noise_tensor_len = noise_batch.shape[1]
        added_tensor_len = added_noise.shape[1]

        pad = (0, abs(noise_tensor_len - added_tensor_len))
        if noise_tensor_len > added_tensor_len:
            added_noise = torch.nn.functional.pad(added_noise, pad)
            added_lens = added_lens * added_tensor_len / noise_tensor_len
        else:
            noise_batch = torch.nn.functional.pad(noise_batch, pad)
            noise_lens = noise_lens * noise_tensor_len / added_tensor_len

        noise_batch = torch.cat((noise_batch, added_noise))
        noise_lens = torch.cat((noise_lens, added_lens))
        noise_names = noise_names+added_noise_names

        return noise_batch, noise_lens, noise_names

    def _load_noise_batch(self):
        """Load a batch of noises, restarting iteration if necessary."""
        # print('in _load_noise_batch')
        try:
            # Don't necessarily know the key
            noises, lens, noise_name = next(self.noise_data)
        except StopIteration:
            self.noise_data = iter(self.data_loader)
            noises, lens, noise_name = next(self.noise_data)
        # print('_load_noise_batch')
        return noises, lens, noise_name







class my_noise_dataset(torch.utils.data.Dataset):
    def __init__(self, noise_dir):
        super(my_noise_dataset, self).__init__()
        self.noise_dir = noise_dir
        # self.noise_dir2data = {}
        self.index2noise_dir = {}
        # self.noise_len = []
        count = 0
        self.sr = 16000
        noise_dirs = [noise_dir]
        if ',' in noise_dir:
            noise_dirs = noise_dir.split(',')
        for noise_dir in noise_dirs:
            print('read from {}'.format(noise_dir))
            for root, dirs, files in os.walk(noise_dir):
                for f in files:
                    path = os.path.join(root, f)
                    if path.endswith('.wav'):
                        self.index2noise_dir[count] = path
                        count += 1
                        if count % 10000 == 0:
                            print('process {}'.format(count))
                        # sr, data = wavfile.read(path)
                        # self.noise_dir2data[path] = data
                        # self.noise_len.append(len(data))
            # print('noise read done')

    def __getitem__(self, index):
        noise_dir = self.index2noise_dir[index]
        # print('{} {}'.format(index,noise_dir))
        # data = self.noise_dir2data[noise_dir]
        # length = len(data)
        return noise_dir

    def __len__(self):
        return len(self.index2noise_dir)

if __name__ == '__main__':
    dataset = my_noise_dataset(os.path.join('/ssd1','qiang','musan','free-sound'))
    a = AddNoise('/ssd1/qiang/musan/free-sound', num_workers=0, snr_low=0,snr_high=15,pad_noise=True,batch_size=5)
    data_loader = DataLoader(
                    dataset, 
                    batch_size=5, 
                    shuffle=True, 
                    sampler=None, 
                    num_workers=0, 
                    collate_fn=a.Collate_fn, 
                    pin_memory=False, 
                    drop_last=False,
                )

    for data,l,noise_name in data_loader:
        print(noise_name)
        exit()













def compute_amplitude(waveforms, lengths=None, amp_type="avg", scale="linear"):
    """Compute amplitude of a batch of waveforms.
    Arguments
    ---------
    waveform : tensor
        The waveforms used for computing amplitude.
        Shape should be `[time]` or `[batch, time]` or
        `[batch, time, channels]`.
    lengths : tensor
        The lengths of the waveforms excluding the padding.
        Shape should be a single dimension, `[batch]`.
    amp_type : str
        Whether to compute "avg" average or "peak" amplitude.
        Choose between ["avg", "peak"].
    scale : str
        Whether to compute amplitude in "dB" or "linear" scale.
        Choose between ["linear", "dB"].
    Returns
    -------
    The average amplitude of the waveforms.
    Example
    -------
    >>> signal = torch.sin(torch.arange(16000.0)).unsqueeze(0)
    >>> compute_amplitude(signal, signal.size(1))
    tensor([[0.6366]])
    """
    if len(waveforms.shape) == 1:
        waveforms = waveforms.unsqueeze(0)

    assert amp_type in ["avg", "peak", "total"]
    assert scale in ["linear", "dB"]

    if amp_type == "avg":
        if lengths is None:
            out = torch.mean(torch.abs(waveforms), dim=1, keepdim=True)
        else:
            wav_sum = torch.sum(input=torch.abs(waveforms), dim=1)
            out = wav_sum / lengths
    elif amp_type == "peak":
        out = torch.max(torch.abs(waveforms), dim=1, keepdim=True)
    else:
        # if lengths is not None:
        #     for index,wav in enumerate(waveforms):
        #         wavfile.write('/ssd1/qiang/test_aug/{}.wav'.format(index), 16000, wav[:lengths[index]].numpy().astype(np.int16))
        t = waveforms*waveforms
        # print(waveforms)
        # print(t)
        
        if lengths is not None:
            out = []
            temp_wavform = t.tolist()
            temp_length = lengths.tolist()
            for index,wav in enumerate(temp_wavform):
                # wavfile.write('/ssd1/qiang/test_aug/{}.wav'.format(index), 16000, np.array(wav[:temp_length[index]]).astype(np.int16))
                out.append(sum(wav[:temp_length[index]]))
            return torch.tensor(out)
        out = torch.sum(input=t, dim=1, keepdim=True)

    if scale == "linear":
        return out
    elif scale == "dB":
        return torch.clamp(20 * torch.log10(out), min=-80)  # clamp zeros
    else:
        raise NotImplementedError




























def dB_to_amplitude(SNR):
    """Returns the amplitude ratio, converted from decibels.
    Arguments
    ---------
    SNR : float
        The ratio in decibels to convert.
    Example
    -------
    >>> round(dB_to_amplitude(SNR=10), 3)
    3.162
    >>> dB_to_amplitude(SNR=0)
    1.0
    """
    return 10 ** (SNR / 20)



def dB_to_power(SNR):
    """Returns the amplitude ratio, converted from decibels.
    Arguments
    ---------
    SNR : float
        The ratio in decibels to convert.
    Example
    -------
    >>> round(dB_to_amplitude(SNR=10), 3)
    3.162
    >>> dB_to_amplitude(SNR=0)
    1.0
    """
    return 10 ** (SNR / 10)