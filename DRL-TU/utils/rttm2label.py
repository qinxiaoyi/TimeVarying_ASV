'''
Generate label file from rttm.
input: rttm, subsegments: [subutt, reco, start, end]
output: subutt2label: [subutt, speaker id]

The disadvantage of this labeling method is that each label can only have a singel label.
'''

import os
import argparse
import numpy as np
from collections import defaultdict

def read_rttm(rttm_file):
    rttm = defaultdict(list)
    for line in open(rttm_file):
        _, reco,channel, start, dur, _, _, spk, _, _ = line.split()
        rttm[reco].append([channel, float(start), float(start) + float(dur), spk])
    return rttm

def read_segments(segments_file):
    segments = defaultdict(list)
    for line in open(segments_file):
        utt, reco, start, end = line.split()
        segments[reco].append([utt, float(start), float(end)])
    return segments

# set args
parser = argparse.ArgumentParser(description='')
parser.add_argument('--rttm', default=None, type=str,
                  help='input rttm file')
parser.add_argument('--segments', default=None, type=str,
                  help='input subsegments file')
parser.add_argument('--frame_length', default=0.025, type=float)
parser.add_argument('--frame_shift', default=0.01, type=float)

parser.add_argument('--write_npy', default=False, type=bool)

args = parser.parse_args()

# read file
rttm = read_rttm(args.rttm)
subsegments = read_segments(args.segments)

# generate label
for reco in sorted(rttm.keys()):
    st       = [i[1] for i in rttm[reco]]
    et       = [i[2] for i in rttm[reco]]
    spks     = sorted(list(set([i[3] for i in rttm[reco]])))
    spk2int  = {spk:i for i, spk in enumerate(spks)}
    n_spks   = len(spks)
    n_frames = int(max(et) / args.frame_shift)

    labels = np.zeros(shape=(n_frames, n_spks), dtype='int') # labels: n_frames x n_spks
    for channel, start, end, spk in rttm[reco]:
        s = round(start / args.frame_shift)
        e = round(end / args.frame_shift)
        labels[s:e, spk2int[spk]] = 1

    for utt, start, end in subsegments[reco]:
        s = round(start / args.frame_shift)
        e = round(end / args.frame_shift)
        duration   = labels[s:e].sum(axis=0)
        target_spk = np.argmax(duration)
        print('%s\t%s' % (utt, target_spk))

