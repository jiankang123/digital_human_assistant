import sys
import os

def main():
    trans = sys.argv[1]
    target_labels = sys.argv[2].split(',')
    new_trans1 = sys.argv[3]
    new_trans2 = sys.argv[4]
    count = {}
    for label in target_labels:
        count[label] = 0
    fout1 = open(new_trans1, 'w', encoding='utf-8')
    fout2 = open(new_trans2, 'w', encoding='utf-8')
    with open(trans, 'r', encoding='utf-8') as fin:
        for line in fin:
            if '\t' in line:
                text = line.strip().split('\t')[1]
            else:
                text = line.strip().split(' ', 1)[1]
            text = text.upper()
            if text in target_labels:
                fout1.write(line)
                count[text] += 1
            else:
                flag = False
                for k in target_labels:
                    if k in text:
                        flag = True
                        break
                if flag:
                    count[k] += 1
                    fout1.write(line.split('\t')[0]+'\t'+k+'\n')
                else:
                    fout2.write(line.split('\t')[0]+'\tGARBAGE\n')
                    print("{} does not have target label".format(text))
    fout1.close()
    fout2.close()
    print(count)
    print("All done")


if __name__ == "__main__":
    main()