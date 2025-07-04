import os
import sys


if __name__ == '__main__':
    wav_dir = sys.argv[1]
    lable = sys.argv[2]
    trans_path = os.path.join(wav_dir, 'trans.txt')
    with open(trans_path, 'w', encoding='utf-8') as w:
        for root, dirs, files in os.walk(wav_dir):
            for f in files:
                if f.endswith('.wav'):
                    wav_path = os.path.abspath(os.path.join(root,f))
                    # label = os.path.basename(os.path.dirname(wav_path))
                    #label = f.split('_')[3]
                    label = lable
                    #label = "HEYSIRI"
                    w.write('{}\t{}\n'.format(wav_path, label))

    print(f"done {trans_path}")

