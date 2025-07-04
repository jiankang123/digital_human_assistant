import torch
import numpy as np
from feat_cpp.feature import FilterbankExtractor
from feat import compute_fbank

FILTERBANK_LEN = 23

class kws_dataset(torch.utils.data.Dataset):
    # @profile
    def __init__(self, transcript, hotwords, has_boundary=False, logging=True):
        # transcript: string 
        # hotwords: dict
        super(kws_dataset, self).__init__()
        self.samples = []
        self.has_boundary = has_boundary
        if logging:
            print("hotwords:", hotwords)
        with open(transcript, 'r', encoding='utf-8') as trans:
            for line in trans:
                if not has_boundary:
                    ptfile = line.strip().split('\t')[0]
                    text = line.strip().split('\t')[1]
                else:
                    arr = line.strip().split('\t')
                    if len(arr) == 4:
                        ptfile, text, begin, end = arr
                        begins = [int(x) for x in begin.split(',')]
                        ends = [int(x) for x in end.split(',')]
                    elif len(arr) == 2:
                        ptfile, text = arr
                        begins = ends = [-1] * len(text.split(','))
                    else:
                        print(f"{line} has problem")
                texts = text.upper().split(',')
                labels = []
                for i, t in enumerate(texts):
                    if not t in hotwords.keys():
                        if logging:
                            print("{} is not a target hotword,".format(t) 
                                  + "treat it as GARBAGE")
                        labels.append(hotwords["GARBAGE"])
                        if has_boundary:
                            begins[i] = -1
                            ends[i] = -1
                    else:
                        labels.append(hotwords[t])
                if not has_boundary:
                    self.samples.append((ptfile, labels))
                else:
                    self.samples.append((ptfile, labels, begins, ends))

    # @profile
    def __getitem__(self, index):
        if not self.has_boundary:
            ptfile, labels = self.samples[index]
        else:
            ptfile, labels, begins, ends = self.samples[index]
        fbank, seq_len = torch.load(ptfile)
        label = torch.tensor(np.array(labels))
        if not self.has_boundary:
            return fbank, seq_len, label

        else:
            return fbank, seq_len, label, begins, ends

    # @profile
    def __len__(self):
        return len(self.samples)


class wav_dataset(torch.utils.data.Dataset):
    def __init__(self, trans_path, has_boundary=False):
        super(wav_dataset, self).__init__()
        self.trans_path = trans_path
        self.has_boundary = has_boundary
        with open(trans_path, 'r', encoding='utf-8') as fin:
            self.lines = fin.readlines()
            
    def __getitem__(self, index):
        wav_path=''
        if '\t' in self.lines[index]:
            if not self.has_boundary:
                wav_path, text = self.lines[index].strip().split('\t')
            else:
                arr = self.lines[index].strip().split('\t')
                if len(arr) == 4:
                    wav_path, text, begin, end = arr
                elif len(arr) == 2:
                    wav_path, text = arr
                    begin = end = str(0)
                else:
                    print(self.lines[index],"There is a problem with this line")
        else:
            wav_path, text = self.lines[index].strip().split(' ', 1)
            if len(self.lines[index].strip().split(' ')) < 2:
                raise ValueError('line format is wrong: ' + self.lines[index])
            if self.has_boundary:
                begin = end = str(0)
        try:
            #print(f"processing {wav_path}")
            with open(wav_path, 'rb') as f:
                data = f.read()
            #print("processed one wav")
            audio = np.frombuffer(data[44:], dtype=np.int16)
            if len(audio) > 80000:
                audio = audio[:80000]
            if not self.has_boundary:
                return torch.from_numpy(np.array(audio)), text, wav_path
            else:
                end = '0' if end == '-1' else end
                begin = '0' if begin == '-1' else begin
                return torch.from_numpy(np.array(audio)), text, begin, end, wav_path
        except FileNotFoundError as e:
            print(f"文件未找到：{wav_path}, line: {self.lines[index]}")
            print(f"出错的行: {self.lines[index].strip()}")
            raise e
            #return None

    def __len__(self):
        return len(self.lines)

# 将多个wav组成一个batch
def wav_Collate_fn(batch_data):
    has_boundary = len(batch_data[0]) == 5
    batch_size = len(batch_data)
    lengths = [len(d[0]) for d in batch_data]
    lengths = torch.tensor(np.array(lengths))
    batch_audio = torch.zeros((batch_size, lengths.max()))
    batch_text = []
    batch_path = []
    if has_boundary:
        batch_begin = []
        batch_end = []
    for index, item in enumerate(batch_data):
        batch_audio[index][:lengths[index]] = item[0]
        if ',' in item[1]:
            print(f'{item[-1]} text {item[1]} has comma, auto remove it')
            item[1] = item[1].replace(',' '')
        batch_text.append(item[1])
        if has_boundary:
            batch_begin.append(item[2])
            batch_end.append(item[3])
        batch_path.append(item[-1])
    if has_boundary:
        return batch_audio, lengths, batch_text, batch_begin, \
               batch_end, batch_path
    else:
        return batch_audio, lengths, batch_text, batch_path

class hw_dataset(torch.utils.data.Dataset):
    # @profile
    def __init__(self, transcript, hotwords, logging=True):
        # transcript: string 
        # hotwords: dict
        super(hw_dataset, self).__init__()
        self.samples = []
        if logging:
            print("hotwords:", hotwords)
        with open(transcript, 'r', encoding='utf-8') as trans:
            for line in trans:
                if '\t' in line:
                    wav_path, text = line.strip().split('\t')
                else:
                    wav_path, text = line.strip().split(' ', 1)
                text = text.upper()
                if text not in hotwords.keys():
                    if logging:
                        print("{} is not a target hotword,".format(text) 
                                  + "treat it as GARBAGE")
                    text = "GARBAGE"
                self.samples.append((wav_path, hotwords[text]))

    # @profile
    def __getitem__(self, index):
        wav_path, label = self.samples[index]
        sr, data = wavfile.read(wav_path)
        return torch.tensor(data), torch.tensor(label)

    # @profile
    def __len__(self):
        return len(self.samples)




def trans_test_Collate_fn(batch_data):
    path = []
    batch_size = len(batch_data)
    lengths = [len(d[0]) for d in batch_data]
    lengths = torch.tensor(np.array(lengths))
    batch_wav = torch.zeros((batch_size, lengths.max()))
    label = torch.zeros(batch_size)

    for index, item in enumerate(batch_data):
        label[index] = item[1]
        batch_wav[index][:lengths[index]] = item[0]
        # lengths.append(int(math.ceil(len(item[0])/160)))
        if len(batch_data[0]) == 3:
            path.append(item[2])

    if len(batch_data[0]) == 3:
        return batch_wav, lengths, label, path
    else:
        return batch_wav, lengths, label


class trans_test_dataset(torch.utils.data.Dataset):
    def __init__(self, trans_path, hotwords_dict, logging=False):
        super(trans_test_dataset, self).__init__()
        self.trans_path = trans_path

        self.wav_path_list = []
        self.label_list = []

        with open(trans_path, 'r', encoding='utf-8') as fin:
            for line in fin:
                if '\t' in line:
                    wav_path = line.strip().split('\t')[0] 
                    text = line.strip().split('\t')[1]
                else:
                    wav_path, text = line.strip().split(' ', 1)
                text = text.upper()
                if text in hotwords_dict.keys():
                    label = hotwords_dict[text]
                else:
                    f = False
                    for k in hotwords_dict.keys():
                        if k in text:
                            f = True
                            break
                    if f:
                        label = hotwords_dict[k]
                    else:
                        if logging:
                            print("{} is not a target hotword,".format(text)
                                  + "change it as garbage")
                        label = hotwords_dict["GARBAGE"]
                self.wav_path_list.append(wav_path)
                self.label_list.append(label)

    def __getitem__(self, index):
        # sr, data = wavfile.read(self.wav_path_list[index], mmap=True)
        data = open(self.wav_path_list[index], 'rb').read()
        audio = np.frombuffer(data[44:], dtype=np.int16)
        label = self.label_list[index]
        return torch.from_numpy(np.array(audio)), label, self.wav_path_list[index]

    def __len__(self):
        return len(self.wav_path_list)



class wav_feat_dataset(torch.utils.data.Dataset):
    def __init__(self, trans_path, hotwords_dict, feat_type="c_fbank", logging=False):
        super(wav_feat_dataset, self).__init__()
        self.trans_path = trans_path
        self.feat_type = feat_type
        if logging:
            print(f"Using {feat_type} as feature")
        if self.feat_type == "cpp_fbank":
            self.fbank_extractor = FilterbankExtractor()
        
        self.wav_path_list = []
        self.label_list = []

        with open(trans_path, 'r', encoding='utf-8') as fin:
            for line in fin:
                if '\t' in line:
                    wav_path = line.strip().split('\t')[0] 
                    text = line.strip().split('\t')[1]
                else:
                    wav_path, text = line.strip().split(' ', 1)
                text = text.upper()
                if text in hotwords_dict.keys():
                    label = hotwords_dict[text]
                else:
                    f = False
                    for k in hotwords_dict.keys():
                        if k in text:
                            f = True
                            break
                    if f:
                        label = hotwords_dict[k]
                    else:
                        if logging:
                            print("{} is not a target hotword,".format(text)
                                  + "change it as garbage")
                        label = hotwords_dict["GARBAGE"]
                self.wav_path_list.append(wav_path)
                self.label_list.append(label)

    def __getitem__(self, index):
        wav_path = self.wav_path_list[index]
        if self.feat_type == "c_fbank":
            fbank = compute_fbank(wav_path)
        elif self.feat_type == "cpp_fbank":
            fbank = self.fbank_extractor.extract(wav_path)
        else:
            print("Error feat type")
        label = self.label_list[index]
        fbank = torch.tensor(fbank, dtype=torch.float32)
        return fbank, label, wav_path

    def __len__(self):
        return len(self.wav_path_list)


def wav_feat_Collate_fn(batch_data):
    path = []
    batch_size = len(batch_data)
    lengths = [d[0].shape[0] for d in batch_data]
    lengths = torch.tensor(np.array(lengths))
    batch_feat = torch.zeros((batch_size, lengths.max(), FILTERBANK_LEN))
    label = torch.zeros(batch_size)

    for index, item in enumerate(batch_data):
        label[index] = item[1]
        # print(batch_feat[index][:lengths[index]].shape)
        # print(item[0].shape)
        batch_feat[index][:lengths[index]] = item[0]
        if len(batch_data[0]) == 3:
            path.append(item[2])

    if len(batch_data[0]) == 3:
        return batch_feat, lengths, label, path
    else:
        return batch_feat, lengths, label




# # @profile
# def main():
#     transcript = '/home/kai.zhou2221/workspace/hotword/speech/hotword/vivo_cmds/exp_mini/dev.trans'
#     hotwords = {
#             "GARBAGE": 0,
#             "快进一首": 1,
#             "开始播放": 2,
#             "暂停播放": 3,
#             "停止播放": 3,
#             "接听电话": 4,
#             "挂断电话": 5,
#             "增大音量": 6,
#             "减小音量": 7,
#             "后退一首": 8
#         }
#     dataset = kws_dataset(transcript, hotwords)
#     dataloader = torch.utils.data.DataLoader(
#         dataset=dataset, 
#         batch_size=32, 
#         shuffle=False, 
#         sampler=None, 
#         batch_sampler=None, 
#         num_workers=0, 
#         collate_fn=kws_collate_fn, 
#         pin_memory=False, 
#         drop_last=False
#     )

#     count = 0
#     for data in dataloader:
#         # print(data)
#         # kws_collate_fn(data)
#         print(data[0].shape, data[1].shape)
#         count += 1
#         if count == 50:
#             break


if __name__ == "__main__":
    # main()

    transcript = '/home/kai.zhou2221/workspace/hotword/speech/hotword/vivo_cmds/exp_mini/block_dev.trans'
    hotwords = {
            "GARBAGE": 0,
            "快进一首": 1,
            "开始播放": 2,
            "暂停播放": 3,
            "停止播放": 3,
            "接听电话": 4,
            "挂断电话": 5,
            "增大音量": 6,
            "减小音量": 7,
            "后退一首": 8
        }
    dataset = kws_dataset(transcript, hotwords)
    dataloader = torch.utils.data.DataLoader(
        dataset=dataset, 
        batch_size=1, 
        shuffle=False, 
        sampler=None, 
        batch_sampler=None, 
        num_workers=0, 
        collate_fn=None, 
        pin_memory=False, 
        drop_last=False
    )
    for data in dataloader:
        print(data[0].shape)
        print(data[1].shape)
        print(data[0], data[1])