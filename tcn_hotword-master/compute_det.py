import sys
import os
import json
import numpy as np

def load_label_and_score(keyword, label_file, score_file):
    # score_table: {uttid: [keywordlist]}
    score_table = {}
    with open(score_file, 'r', encoding='utf-8') as fin:
        for line in fin:
            arr = line.strip().split()
            key = arr[0]
            current_keyword = arr[1]
            str_list = arr[2:]
            if int(current_keyword) == keyword:
                scores = list(map(float, str_list))
                if key not in score_table:
                    score_table.update({key: scores})
    keyword_table = {}
    filler_table = {}
    filler_duration = 0.0
    with open(label_file, 'r', encoding='utf8') as fin:
        for line in fin:
            obj = json.loads(line.strip())
            assert 'key' in obj
            assert 'txt' in obj
            assert 'duration' in obj
            key = obj['key']
            index = obj['txt']
            duration = obj['duration']
            assert key in score_table
            if index in hotword_dict.keys():
                if hotword_dict[index] == keyword:
                    keyword_table[key] = score_table[key]
                else:
                    filler_table[key] = score_table[key]
                    filler_duration += duration
            else:
                filler_table[key] = score_table[key]
                filler_duration += duration
    return keyword_table, filler_table, filler_duration


def is_contain(tr, n):
    for i in range(tr.shape[0]-n):
        if tr[i:i+n].sum() == n:
            return True
    return False



if __name__ == '__main__':
    keyword = int(sys.argv[1])
    test_data = sys.argv[2]
    score_file = sys.argv[3]
    hotwords = sys.argv[4].split(',')

    stats_file = os.path.join(os.path.dirname(score_file), str(keyword)+'.txt')

    hotword_dict = {"GARBAGE": 0}
    for index, hotword in enumerate(hotwords):
        hotword_dict[hotword] = index + 1
        
    window_shift = 60
    step = 0.01
    keyword_table, filler_table, filler_duration = load_label_and_score(
        keyword, test_data, score_file)
    print('Filler total duration Hours: {}'.format(filler_duration / 3600.0))
    with open(stats_file, 'w', encoding='utf8') as fout:
        keyword_index = int(keyword)
        threshold = 0.5
        while threshold <= 1.0:
            num_false_reject = 0
            # transverse the all keyword_table
            for key, score_list in keyword_table.items():
                # computer positive test sample, use the max score of list.
                score = max(score_list)
                if float(score) < threshold:
                    num_false_reject += 1
                
            num_false_alarm = 0
            # transverse the all filler_table
            for key, score_list in filler_table.items():
                score = max(score_list)
                if float(score) < threshold:
                    continue
                i = 0
                while i < len(score_list):
                    if score_list[i] >= threshold:
                        num_false_alarm += 1
                        i += window_shift
                    else:
                        i += 1
            if len(keyword_table) != 0:
                false_reject_rate = num_false_reject / len(keyword_table)
            else:
                false_reject_rate = -1
            num_false_alarm = max(num_false_alarm, 1e-6)
            if filler_duration != 0:
                false_alarm_per_hour = num_false_alarm / (filler_duration / 3600.0)
            else:
                false_alarm_per_hour = -1
            fout.write('{:.6f} {:.6f} {:.6f}\n'.format(threshold,
                                                       false_alarm_per_hour,
                                                       false_reject_rate))
            threshold += step

    print(f"{stats_file} done")
