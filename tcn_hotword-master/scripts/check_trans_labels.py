import sys
import os
from tqdm import tqdm

def main():
    trans = sys.argv[1]
    labels = sys.argv[2].split(',')
    if len(sys.argv) > 3:
        new_trans = sys.argv[3]
        fout = open(new_trans, 'w', encoding='utf-8')
    with open(trans, 'r', encoding='utf-8') as fin:
        for line in tqdm(fin):
            if '\t' in line:
                file_path, text = line.strip().split('\t')
            else:
                file_path, text = line.strip().split(' ', 1)
            valid = True
            if text in labels:
                valid = False
            # for w in labels:
            #     if w.upper() in text.upper():
            #         valid = False
            #         print("{} label {} has word {}!".format(file_path, text, w))
            #         break
            if not valid and len(sys.argv) > 3:
                fout.write(line)
    if len(sys.argv) > 3:
        fout.close()

    print("All done")


if __name__ == "__main__":
    main()