import sys
import os
import json
import numpy as np

def load_score(keyword, score_file):
    with open(score_file, 'r', encoding='utf-8') as fin:
        postive_list = []
        negative_list = []
        filler_duration = 0.0
        for line in fin:
            arr = line.strip().split()
            label = int(arr[0])
            current_keyword = int(arr[1])
            str_list = arr[2:]
            if current_keyword == keyword:
                scores = list(map(float, str_list))
                if label == keyword:
                    postive_list.append(scores)
                else:
                    negative_list.append(scores)
                    filler_duration += len(scores) / 100
    return postive_list, negative_list, filler_duration


def is_contain(tr, n):
    for i in range(tr.shape[0]-n):
        if tr[i:i+n].sum() == n:
            return True
    return False


if __name__ == '__main__':
    keyword = int(sys.argv[1])
    score_file = sys.argv[2]
    hotwords = sys.argv[3].split(',')
    sustain_num = int(sys.argv[4])

    stats_file = os.path.join(os.path.dirname(score_file), 
                              str(keyword)+ '_' + str(sustain_num) +'.txt')

    hotword_dict = {"GARBAGE": 0}
    for index, hotword in enumerate(hotwords):
        hotword_dict[hotword] = index + 1
        
    window_shift = 60
    step = 0.01

    postive_list, negative_list, filler_duration = load_score(keyword, score_file)

    print('Filler total duration Hours: {}'.format(filler_duration / 3600.0))
    with open(stats_file, 'w', encoding='utf8') as fout:
        keyword_index = int(keyword)
        threshold = 0.5
        while threshold <= 1.0:
            num_false_reject = 0
            for score_list in postive_list:
                score = max(score_list)
                if float(score) < threshold:
                    num_false_reject += 1
                else:
                    tr = (np.array(score_list) >= threshold).astype(np.int32)
                    if is_contain(tr, sustain_num):
                        continue
                    else:
                        num_false_reject += 1

            num_false_alarm = 0
            for score_list in negative_list:
                score = max(score_list)
                if float(score) < threshold:
                    continue
                else:
                    tr = (np.array(score_list) >= threshold).astype(np.int32)
                    if is_contain(tr, sustain_num):
                        num_false_alarm += 1
                    else:
                        continue
                    # i = 0
                    # while i < len(score_list):
                    #     if score_list[i] >= threshold:
                    #         num_false_alarm += 1
                    #         i += window_shift
                    #     else:
                    #         i += 1
            if len(postive_list) != 0:
                false_reject_rate = num_false_reject / len(postive_list)
            else:
                false_reject_rate = -1
            # num_false_alarm = max(num_false_alarm, 1e-6)
            if filler_duration != 0:
                false_alarm_per_hour = num_false_alarm / (filler_duration / 3600.0)
            else:
                false_alarm_per_hour = -1
            fout.write('{:.6f} {:.6f} {:.6f}\n'.format(threshold,
                                                       false_alarm_per_hour,
                                                       false_reject_rate))
            threshold += step

    print(f"{stats_file} done")
