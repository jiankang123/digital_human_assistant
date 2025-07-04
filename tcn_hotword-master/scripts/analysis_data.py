import argparse
import random
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trans", default="")
    args = parser.parse_args()

    label_dict = {}

    with open(args.trans, 'r', encoding='utf-8') as fin:
        lines = fin.readlines()
    for i in tqdm(range(len(lines))):
        line = lines[i]
        wav_path, text = line.strip().split('\t')[0], line.strip().split('\t')[1]
        if text in label_dict.keys():
            label_dict[text].append(line)
        else:
            label_dict[text] = [line]
            
    print(f"hotwords: {[(k, len(label_dict[k])) for k in label_dict.keys()]}")


if __name__ == "__main__":
    main()

