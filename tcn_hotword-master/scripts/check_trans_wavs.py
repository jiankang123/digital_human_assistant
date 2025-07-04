import sys
import os
from tqdm import tqdm

def main():
    trans = sys.argv[1]
    if len(sys.argv) > 2:
        new_trans = sys.argv[2]
        fout = open(new_trans, 'w', encoding='utf-8')
    with open(trans, 'r', encoding='utf-8') as fin:
        lines = fin.readlines()
        for i in tqdm(range(len(lines))):
            line = lines[i]
            if '\t' in line:
                arr = line.strip().split('\t')
                if len(arr) == 4:
                    file_path, text, begin, end = arr
                elif len(arr) == 2:
                    file_path, text = arr
                # file_path  = line.strip().split('\t')[0]
            else:
                print("no \\t")
                file_path  = line.strip().split(' ', 1)[0]
            if not os.path.exists(file_path):
                print("{} not exists!".format(file_path))
                continue
            else:
                if len(sys.argv) > 2:
                    fout.write(line)
    print("All done")
    if len(sys.argv) > 2:
        fout.close()

if __name__ == "__main__":
    main()