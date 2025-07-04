# Copyright 2019 Mobvoi Inc. All Rights Reserved.
# Author: fangjun.kuang@mobvoi.com (Fangjun Kuang)

import numpy as np


def float_to_q31(number, frac):
    a = np.round(np.float32(number) * (1 << int(frac)))
    if a > np.iinfo(np.int32).max:
        a = np.iinfo(np.int32).max
    if a < np.iinfo(np.int32).min:
        a = np.iinfo(np.int32).min
    return np.int32(a)


def q31_to_float(number, frac):
    return np.float32(np.int32(number)) / (1 << frac)


def float_to_q15(number, frac):
    a = np.round(np.float32(number) * (1 << int(frac)))
    if a > np.iinfo(np.int16).max:
        a = np.iinfo(np.int16).max
    if a < np.iinfo(np.int16).min:
        a = np.iinfo(np.int16).min
    return np.int16(a)


def q15_to_float(number, frac):
    return np.float32(np.int16(number)) / (1 << frac)

def q15_weight_to_float(weight, frac):
    qw = []
    for wi in weight:
        qw.append(q15_to_float(wi, frac))
    qweight = np.array(qw)
    return qweight


def float_to_q7(number, frac):
    a = np.round(np.float32(number) * (1 << int(frac)))
    if a > 127:
        a = 127
    if a < -128:
        a = -128
    a = np.int8(a)
    return a


def q7_to_float(number, frac):
    return np.float32(np.int8(number)) / (1 << frac)


def q31_to_q15(v, q31_frac, q15_frac):
    assert q31_frac > q15_frac
    v = np.int32(v) >> (q31_frac - q15_frac)
    if v > np.iinfo(np.int16).max:
        v = np.iinfo(np.int16).max
    if v < np.iinfo(np.int16).min:
        v = np.iinfo(np.int16).min
    return np.int16(v)


def get_min_max_value(value):
    min_v = np.amin(value)
    max_v = np.amax(value)
    #  print("min max: ", min_v, max_v)
    return min_v, max_v


# 计算8bit定点化的frac
def get_8bit_frac(w):
    min_v, max_v = get_min_max_value(w)
    assert -128 <= min_v <= 127
    assert -128 <= max_v <= 127
    v = max(abs(min_v), abs(max_v))
    '''
    bits = [
        (1, 7),  # q0.7
        (2, 6),  # q1.6
        (4, 5),  # q2.5
        (8, 4),  # q3.4
        (16, 3),  #q4.3
        (32, 2),  #q5.2
        (64, 1),  #q6.1
        (128, 0),  #q7.0
    ]
    '''
    bits = list()
    for i in reversed(range(0, 8)):
        bits.append((1 << (7 - i), i))

    res = 7
    for i in bits:
        if v <= i[0]:
            res = i[1]
            break
    return res


# 计算32bit定点化的frac
def get_32bit_frac(b):
    min_v, max_v = get_min_max_value(b)
    assert np.iinfo(np.int32).min <= min_v <= np.iinfo(np.int32).max
    assert np.iinfo(np.int32).min <= max_v <= np.iinfo(np.int32).max
    v = max(abs(min_v), abs(max_v))
    '''
    bits = [
        (1, 31),  # q0.31
        (2, 30),  # q1.30
        (4, 29),  # q2.29
        ...
    ]
    '''
    bits = list()
    for i in reversed(range(0, 32)):
        bits.append((1 << (31 - i), i))

    res = 31
    for i in bits:
        if v <= i[0]:
            res = i[1]
            break
    return res
