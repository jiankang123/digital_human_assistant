import os
import sys


if __name__ == '__main__':
    wav_dir = sys.argv[1]
    pos_path = os.path.join(wav_dir, 'pos.trans')
    neg_path = os.path.join(wav_dir, 'neg.trans')
    with open(pos_path, 'w', encoding='utf-8') as w1:
     with open(neg_path, 'w', encoding='utf-8') as w2:
        for root, dirs, files in os.walk(wav_dir):
            for f in files:
                if f.endswith('.wav'):
                    wav_path = os.path.abspath(os.path.join(root,f))
                    # label = os.path.basename(os.path.dirname(wav_path))
                    label = f.split('_')[3]
                    if label == "haixiaowen":
                        label = "嗨小问"
                        w1.write('{}\t{}\n'.format(wav_path, label))
                    else:
                        label = "garbage"
                        w2.write('{}\t{}\n'.format(wav_path, label))                 
                    

    print(f"done {pos_path} {neg_path}")

