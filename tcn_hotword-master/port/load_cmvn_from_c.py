import argparse
import json
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--h_file", default="")
    parser.add_argument("--cmvn", default="")
    args = parser.parse_args()

    h_file = args.h_file
    cmvn_json = args.cmvn

    params = []
    with open(h_file, 'r', encoding='utf-8') as fin:
        load = False
        for line in fin:
            if line.strip().endswith('{'):
                layer_param = []
                load = True
                continue
            elif line.strip().endswith('};'):
                params.append(layer_param)
                load = False
                continue

            if load:
                for p in line.strip().split(','):
                    if len(p) > 0:
                        if p.endswith('f'):
                            layer_param.append(float(p[:-1]))
                        else:
                            layer_param.append(float(p))

    stat = {
        'utterances': None,
        'frames': None,
        'mean': params[0],
        'scale': params[1]
    }
    with open(cmvn_json, 'w', encoding='utf-8') as f:
        json.dump(stat, f, indent=4)

if __name__ == "__main__":
    main()