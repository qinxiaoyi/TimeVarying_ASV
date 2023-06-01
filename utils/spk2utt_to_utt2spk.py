import os
import argparse

args = argparse.ArgumentParser(description='')
args.add_argument('-i', '--spk2utt', default=None, type=str)
args = args.parse_args()

utt2spk = []
for line in open(args.spk2utt):
    spk, utts = line.split()[0], line.split()[1:]
    utt2spk += [[utt, spk] for utt in utts]

for utt, spk in sorted(utt2spk):
    print(utt, spk)

