import argparse
import random
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trans", default="")
    parser.add_argument("--train", default="")
    parser.add_argument("--test", default="")
    parser.add_argument("--dev", default="")
    parser.add_argument("--words", default=None)
    parser.add_argument("--rate", default="0.1")
    parser.add_argument("--num", default="-1")
    args = parser.parse_args()

    words = args.words.split(',')
    label_dict = {"GARBAGE": []}
    for w in words:
        label_dict[w] = []
    print(label_dict)

    rate = float(args.rate)
    num = int(args.num)
    with open(args.trans, 'r', encoding='utf-8') as fin:
        lines = fin.readlines()
    fout1 = open(args.train, 'w', encoding='utf-8')
    fout2 = open(args.test, 'w', encoding='utf-8')
    fout3 = open(args.dev, 'w', encoding='utf-8')
    for i in tqdm(range(len(lines))):
        line = lines[i]
        if '\t' in line:
            text = line.strip().split('\t')[1].upper()
        else:
            text = line.strip().split(' ')[1].upper()
        if text in label_dict.keys():
            label_dict[text].append(line)
        else:
            flag = False
            for k in label_dict.keys():
                if k in text:
                    label_dict[k].append(line.replace('\t'+text, '\t'+k))
                    flag = True
                    break
            if not flag:
                label_dict["GARBAGE"].append(line.replace('\t'+text, "\tGARBAGE"))

    print(f"hotwords: {[(k, len(label_dict[k])) for k in label_dict.keys()]}")
    train_list = []
    test_list = []
    dev_list = []
    for word in label_dict.keys():
        random.shuffle(label_dict[word])
        if num == -1:
            word_test_list = random.sample(label_dict[word],
                                           int(len(label_dict[word]) * rate))
            word_dev_list = random.sample(label_dict[word],
                                          int(len(label_dict[word]) * rate))
        else:
            word_test_list = random.sample(label_dict[word], num)
            word_dev_list = random.sample(label_dict[word], num)
        word_train_list = list(set(label_dict[word]) - set(word_test_list) - set(word_dev_list))
        train_list += word_train_list
        test_list += word_test_list
        dev_list += word_dev_list

    for line in train_list:
        fout1.write(line)
    for line in test_list:
        fout2.write(line)
    for line in dev_list:
        fout3.write(line)

    fout1.close()
    fout2.close()
    fout3.close()


if __name__ == "__main__":
    main()
