import os, sys
import argparse
import numpy as np
import json
from qformat import *

def get_CMVN(CMVN_json):
    with open(CMVN_json) as f:
        stat = json.load(f)
        mean = np.array(stat['mean'])
        scale = np.array(stat['scale'])
    return mean, scale


def weight2str(w):
    cpp_str = ""
    k = 0
    num_per_line = 6
    for wi in w.reshape(-1):
        if k > 1 and k % num_per_line == 0:
            cpp_str += "\n"
        k += 1
        if w.dtype == np.int32:
            cpp_str += "{:11d},".format(wi)
        else:
            cpp_str += "{:16f},".format(wi)
    return cpp_str


# 32bit定点化
def bias_float2int32(b, frac):
    qb = []
    for bi in b:
        qb.append(float_to_q31(bi, frac))
    return np.array(qb).astype(np.int32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cmvn_json", default="")
    parser.add_argument("--date", default="")
    parser.add_argument("--project", default="")
    parser.add_argument("--name", default="")
    
    args = parser.parse_args()
    
    cmvn_json = args.cmvn_json
    out_dir = os.path.dirname(cmvn_json)
    date = args.date
    name = args.name

    mean, scale = get_CMVN(cmvn_json)
    
    cpp_src = "// Auto generated! Do **NOT** edit!"

    cpp_src += "// Generated from {}\n".format(
        os.path.basename(cmvn_json)
    )
    cpp_src += "// Copyright 2022 Mobvoi Inc. All Rights Reserved.\n"
    cpp_src += "// Author: kai.zhou2221@mobvoi.com (Kai Zhou)\n\n"

    cpp_src += "#ifndef CMVN_DATA_{}_H_\n".format(name)
    cpp_src += "#define CMVN_DATA_{}_H_\n\n".format(name)

    if args.project == "vivo":
        cpp_src += "const float kMean[{}] = {{\n".format(
            mean.reshape(-1).shape[0]
        )
    elif args.project == "insta":
        cpp_src += "static const float kMean[{}] = {{\n".format(
            mean.reshape(-1).shape[0]
        )
    elif args.project == "qcc":
        # 与c代码中feature的frac要一致！
        mean_frac = 24
        mean = bias_float2int32(mean, mean_frac)
        cpp_src += "// q{}.{}\n".format(31 - mean_frac, mean_frac)
        cpp_src += "const int32_t kMean[{}] = {{\n".format(
            mean.reshape(-1).shape[0]
        )
        

    cpp_src += weight2str(mean)
    cpp_src += "\n};\n\n"
    
    if args.project == "vivo":
        cpp_src += "const float kScale[{}] = {{\n".format(
            scale.reshape(-1).shape[0]
        )
    elif args.project == "insta":
        cpp_src += "static const float kScale[{}] = {{\n".format(
            scale.reshape(-1).shape[0]
        )
    elif args.project == "qcc":
        scale_frac = 31
        scale = bias_float2int32(scale, scale_frac)
        cpp_src += "// q{}.{}\n".format(31 - scale_frac, scale_frac)
        cpp_src += "const int32_t kScale[{}] = {{\n".format(
            scale.reshape(-1).shape[0]
        )

    cpp_src += weight2str(scale)
    cpp_src += "\n};\n\n"
    cpp_src += "#endif  // CMVN_DATA_{}_H_\n\n".format(name)
    # save cpp_src
    cpp_src_filename = "{}_mean_scale_{}.h".format(
        date, args.project
    )
    with open(out_dir + "/" + cpp_src_filename, "w") as f:
        f.write(cpp_src[:-1])
    print("write\n  {}\nsuccessfully!".format(cpp_src_filename))


if __name__ == "__main__":
    main()