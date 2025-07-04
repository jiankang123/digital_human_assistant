from add_noise import AddNoise
import torch
import numpy as np
import sys
import os
from scipy.io import wavfile


def Collate_fn(batch_data):
    batch_wav = []
    label = []
    for item in batch_data:
        label.append(item[1])
        # sr, data = wavfile.read(item[0])
        data = open(item[0], 'rb').read()
        audio = np.frombuffer(data[44:], dtype=np.int16)
        batch_wav.append(torch.tensor(audio))
    lengths = [len(x) for x in batch_wav]
    wav_name = [x[0].split('/')[-1] for x in batch_data]
    padded_data = torch.nn.utils.rnn.pad_sequence(batch_wav, batch_first=True,padding_value=0)
    return padded_data,torch.tensor(lengths),wav_name,label


class wav_dataset(torch.utils.data.Dataset):
    def __init__(self, transcript, logging=True):
        super(wav_dataset, self).__init__()
        # self.wav_dir2data = {}
        self.index2wav_dir = {}
        self.index2label = {}
        count = 0
        self.sr = 16000
        with open(transcript, 'r', encoding='utf-8') as trans:
            for line in trans:
                if '\t' in line:
                    wav_file = line.strip().split('\t')[0]
                    label = line.strip().split('\t',1)[1]
                else:
                    wav_file = line.strip().split(' ')[0]
                    label = line.strip().split(' ',1)[1]
                # sr, data = wavfile.read(wav_file)
                # self.wav_dir2data[wav_file] = data
                self.index2wav_dir[count] = wav_file
                self.index2label[count] = label
                # self.wav_len.append(len(data))
                count += 1
        # self.maxlen = max(self.wav_len)
        
    def __getitem__(self, index):
        wav_dir = self.index2wav_dir[index]
        label = self.index2label[index]
        # data = self.wav_dir2data[wav_dir]
        # length = len(data)
        return wav_dir, label
              
    def __len__(self):
        return len(self.index2wav_dir)


if __name__ == '__main__':
    trans_dir = sys.argv[1]
    noise_dir = sys.argv[2]
    out_dir = sys.argv[3]

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    batch_size=512
    num_workers=24
    snr_low=0
    snr_high=1

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
    # for data in dataloader:
        # print(data.shape)
    #     print(data)

    count = 0
    augment = AddNoise(noise_dir, num_workers=num_workers, snr_low=snr_low,snr_high=snr_high,pad_noise=True,batch_size=batch_size)
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    w = open(os.path.join(out_dir,'trans.txt'),'w',encoding='utf-8')
    for data,l,wav_dir,labels in dataloader:
        # data,l,wav_dir = data.cuda(), l.cuda(), wav_dir.cuda()
        aug_wav,noise_names,snr = augment(data,l)
        # aug_wav,noise_names,snr = aug_wav.cpu(),noise_names.cpu(),snr.cpu()
        for wav_name,noise_name,snr, aug, length, label in zip(wav_dir,noise_names,snr,aug_wav, l, labels):
            '''print('wav_name',wav_name)
            print('noise_name',noise_name)
            print('snr',snr)'''
            aug_path = wav_name.split('.wav')[0]+'-'+noise_name.split('.wav')[0]+'-'+'SNR{}'.format(snr)+'.wav'
            print('write {}'.format(aug_path))
            wavfile.write(out_dir+'/{}'.format(aug_path), 16000, aug[:length].numpy().astype(np.int16))
            w.write(os.path.join(out_dir,'{}\t'.format(aug_path))+'{}\n'.format(label))
    w.close()
    print('done')
    # for data,len in dataloader:
