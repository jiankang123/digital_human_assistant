import os, sys
import numpy as np


def main():
    score_dir = sys.argv[1]

    score_txt = os.path.join(score_dir, '../score.txt')
    fout = open(score_txt, 'w', encoding='utf-8')

    for root, dirs, files in os.walk(score_dir):
        for file in files:
            if file.endswith('.txt'):
                key = file[:-4]
                
                file_path = os.path.join(root, file)
                score = []
                with open(file_path, 'r', encoding='utf-8') as fin:
                    for line in fin:
                        score.append(line.strip().split(' '))
                score_list = list(map(list, zip(*score)))
                for r in range(1, len(score_list)):
                    fout.write(key+' '+str(r)+' '+' '.join(score_list[r])+'\n')
    fout.close()
if __name__ == "__main__":
    main()