def findcuelabel(data):
    out = []
    '''for i in range(len(data) - 4):
        if data[i:i + 4] == b'cue ':
            out.append(i)'''
    start = 0
    while 1:
        index = data.find(b'cue ', start)
        if index == -1:
            break
        out.append(index)
        start = index + 1
    return out


def checkcuelabel(data, label):
    i = label
    markers = []
    chunkid = data[i:i + 4]
    assert (chunkid == b'cue ')

    chunksize = int.from_bytes(data[i + 4:i + 8], byteorder='little', signed=False)
    dwcuepoints = int.from_bytes(data[i + 8:i + 12], byteorder='little', signed=False)
    assert (chunksize == dwcuepoints * 24 + 4)
    chunkstart = i + 12
    for j in range(dwcuepoints):
        pointstart = chunkstart + j * 24
        dwidentifier = int.from_bytes(data[pointstart:pointstart + 4], byteorder='little', signed=False)
        assert (dwidentifier == j + 1)
        dwposition = int.from_bytes(data[pointstart + 4:pointstart + 8], byteorder='little', signed=False)
        fccchunk = data[pointstart + 8:pointstart + 12]
        dwsampleoffset = int.from_bytes(data[pointstart + 20:pointstart + 24], byteorder='little', signed=False)
        assert (fccchunk == b'data')
        assert (dwsampleoffset == dwposition)
        markers.append(dwsampleoffset)
    return markers


def exportSampleOffset(wavfilename):
    with open(wavfilename, 'rb') as fp:
        data = fp.read()
    cuelabels = findcuelabel(data)
    out = []
    if not cuelabels:
        return []
    for cuelabel in cuelabels:
        try:
            out = checkcuelabel(data, cuelabel)
        except:
            pass
    return out


if __name__ == '__main__':
    print(exportSampleOffset(r'C:\Users\mobvoi\Desktop\115_male_26_lb.wav'))
