import os
import sys


if __name__ == '__main__':
    wav_dir = sys.argv[1]
    trans_path = sys.argv[2]
    # label = 'GARBAGE'
    label = sys.argv[3]
    w = open(trans_path, 'w', encoding='utf-8')
    for root, dirs, files in os.walk(wav_dir):
        for f in files:
            if f.endswith('.wav'):
                if f.startswith('XVXV'):
                    label = '小V小V'
                elif f.startswith('Jovi'):
                    label = 'HIJOVI'
                path = os.path.abspath(os.path.join(root,f))
                w.write('{}\t{}\n'.format(path, label))

    print(f"done {trans_path}")






