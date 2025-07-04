import sys
import os
import shutil

def main():
    trans = sys.argv[1]
    target_dir = sys.argv[2]

    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    new_trans = os.path.join(target_dir, 'trans.txt')
    fout = open(new_trans, 'w', encoding='utf-8')

    with open(trans, 'r', encoding='utf-8') as fin:
        for line in fin:
            if '\t' in line:
                file_path, text = line.strip().split('\t', 1)
            else:
                file_path, text = line.strip().split(' ', 1)
            file_path = file_path.replace('\\','')
            new_file_path = os.path.join(target_dir, os.path.basename(file_path).replace(' ','').replace(',',''))
            shutil.copy(file_path, new_file_path)
            fout.write(new_file_path + '\t' + text + '\n')

    fout.close()
    print("ALL done")

if __name__ == "__main__":
    main()