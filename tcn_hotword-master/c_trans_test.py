import os
import sys
import subprocess
from roc import compute_roc
from anlaysis_confidence import compute_acc


def wav_test(exe_path, wav_path, label, confidence_path):
    cmd = "{} {} {} {}".format(exe_path, wav_path, label, confidence_path)
    exe_out = subprocess.check_output(cmd, shell=True)


def main():
    exe_path = sys.argv[1]
    out_dir = sys.argv[2]
    trans = sys.argv[4]
    hotwords_str = sys.argv[3]
    

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # 将热词组织成字典，key为热词，value为类别
    hotwords_dict = {}
    hotwords = ["GARBAGE"] + hotwords_str.split(',')
    num_classes = len(hotwords)
    for index, hotword in enumerate(hotwords):
        if '/' in hotword:
            hotword_arr = hotword.split('/')
            for h in hotword_arr:
                hotwords_dict[h] = index
        else:
            hotwords_dict[hotword] = index
    print("hotwords:", hotwords_dict)

    confidence_list = []
    with open(trans, 'r', encoding='utf-8') as fin:
        for line in fin:
            if '\t' in line:
                wav_path, text = line.strip().split('\t')
            else:
                wav_path, text = line.strip().split(' ', 1)
            if not text in hotwords_dict.keys():
                print("{} is not a target hotword, change it as garbage".format(text))
                label = hotwords_dict["GARBAGE"]
            else:
                label = hotwords_dict[text]
            wav_name = os.path.basename(wav_path)
            confidence_path = os.path.join(out_dir, wav_name.replace(".wav", ".txt"))
            wav_test(exe_path, wav_path, label, confidence_path)
            confidence_list.append(confidence_path)
    confidence_ark = os.path.join(out_dir, "confidence.ark")

    # if os.path.exists(confidence_ark):
    #     os.remove(confidence_ark)
    # subprocess.check_output("cat {} > {}".format(" ".join(confidence_list), confidence_ark), shell=True)
    fout = open(confidence_ark, 'w', encoding='utf-8')
    for f in confidence_list:
        fout.write(open(f, 'r', encoding='utf-8').read())
    compute_roc(out_dir, num_classes)
    compute_acc(out_dir)


if __name__ == "__main__":
    main()