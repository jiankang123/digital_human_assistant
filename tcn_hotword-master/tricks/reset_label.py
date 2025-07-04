from add_noise import AddNoise
import torch
import numpy as np
import sys
from scipy.io import wavfile


def Collate_fn(batch_data):
    batch_wav = []
    wav_data = []
    for item in batch_data:
        sr, data = wavfile.read(item)
        batch_wav.append((torch.tensor(data), item.split('/')[-1]))
        wav_data.append(torch.tensor(data))
    batch_wav.sort(key=lambda x: len(x[0]),reverse=True)
    sorted_wav_name = [x[1] for x in batch_wav]
    lengths = [len(x) for x in batch_wav]
    maxlen = lengths[0]
    # data_len = [x[1] for x in batch_data]
    # data = [x[0] for x in batch_data]
    wav_name = [x.split('/')[-1] for x in batch_data]
    padded_data = torch.nn.utils.rnn.pad_sequence(wav_data, batch_first=True,padding_value=0)
    return padded_data,torch.tensor(lengths),wav_name, sorted_wav_name


class wav_dataset(torch.utils.data.Dataset):
    def __init__(self, transcript, logging=True):
        super(wav_dataset, self).__init__()
        # self.wav_dir2data = {}
        self.index2wav_dir = {}
        # self.wav_len = []
        count = 0
        self.sr = 16000
        with open(transcript, 'r', encoding='utf-8') as trans:
            for line in trans:
                if '\t' in line:
                    wav_file = line.strip().split('\t')[0]
                else:
                    wav_file = line.strip().split(' ')[0]
                # sr, data = wavfile.read(wav_file)
                # self.wav_dir2data[wav_file] = data
                self.index2wav_dir[count] = wav_file
                # self.wav_len.append(len(data))
                count += 1
        # self.maxlen = max(self.wav_len)
        
    def __getitem__(self, index):
        wav_dir = self.index2wav_dir[index]
        # data = self.wav_dir2data[wav_dir]
        # length = len(data)
        return wav_dir
              
    def __len__(self):
        return len(self.index2wav_dir)


if __name__ == '__main__':
    trans_dir = sys.argv[1]
    # noise_dir = sys.argv[2]
    # out_dir = sys.argv[3]

    batch_size=512
    num_workers=8
    snr_low=0
    snr_high=20

    dataset = wav_dataset(trans_dir)
    dataloader = torch.utils.data.DataLoader(
        dataset=dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        sampler=None, 
        batch_sampler=None, 
        num_workers=num_workers, 
        pin_memory=False,
        drop_last=False,
        collate_fn=Collate_fn
    )
    w = open('map_label.txt','w')
    count = 0
    for data,l,ori_name,sorted_name in dataloader:
        print('batch {}'.format(count))
        count += 1
        for index,item in enumerate(ori_name):
            w.write('{}\t{}\n'.format(item,sorted_name[index]))
        # print(ori_name)
        # print(sorted_name)

    """count = 0
    augment = AddNoise(noise_dir, num_workers=num_workers, snr_low=snr_low,snr_high=snr_high,pad_noise=True,batch_size=batch_size)
        
    for data,l,wav_dir in dataloader:
        # data,l,wav_dir = data.cuda(), l.cuda(), wav_dir.cuda()
        aug_wav,noise_names,snr = augment(data,l)
        # aug_wav,noise_names,snr = aug_wav.cpu(),noise_names.cpu(),snr.cpu()
        for wav_name,noise_name,snr, aug, length in zip(wav_dir,noise_names,snr,aug_wav, l):
            '''print('wav_name',wav_name)
            print('noise_name',noise_name)
            print('snr',snr)'''
            aug_path = wav_name.split('.wav')[0]+'-'+noise_name.split('.wav')[0]+'-'+'SNR{}'.format(snr)+'.wav'
            print('write {}'.format(aug_path))
            wavfile.write(out_dir+'/{}'.format(aug_path), 16000, aug[:length].numpy().astype(np.int16))
    print('done')
    # for data,len in dataloader:"""