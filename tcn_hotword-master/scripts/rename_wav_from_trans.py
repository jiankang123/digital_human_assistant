import sys, os


def main():
    trans_in = sys.argv[1]
    new_lines = []
    with open(trans_in, 'r', encoding='utf-8') as fin:
        for line in fin:
            wav_path, text = line.strip().split('\t')
            new_path = wav_path.replace(' ', '')
            os.rename(wav_path, new_path)
            new_lines.append(new_path+'\t'+text+'\n')

    with open(trans_in, 'w', encoding='utf-8') as fout:
        for line in new_lines:
            fout.write(line)


if __name__ == "__main__":
    main()
