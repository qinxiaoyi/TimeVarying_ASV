'''
convert segments into subsegments with a fixed length
'''

import os
import argparse

# set args
parser = argparse.ArgumentParser(description='')
parser.add_argument('-i', '--input', default=None, type=str,
                  help='input segments file')
parser.add_argument('--win_length', default=1.5, type=float,
                  help='the fixed length of subsegments (in seconds)')
parser.add_argument('--win_shift', default=0.75, type=float,
                  help='the shift length of subsegments (in seconds)')
parser.add_argument('--min_length', default=0.5, type=float,
                  help='the minimal length of subsegments (in seconds)')
parser.add_argument('--constant_duration', default=True, type=bool)
args = parser.parse_args()

for line in open(args.input):
    utt, reco, st, et = line.split()
    st, et = float(st), float(et)

    subst, subet = 0, et - st
    while st + args.win_length < et:
        subutt = "{utt}-{s:08d}-{e:08d}".format(utt=utt, s=round(100 * subst), e=round(100 * (subst + args.win_length)))
        print('%s\t%s\t%.2lf\t%.2lf' % (subutt, reco, st, st + args.win_length))
        st += args.win_shift
        subst += args.win_shift

    if args.constant_duration:
        if subet > args.win_length:
            subst = min(subet - args.win_length, subst)
            st = min(et - args.win_length, st)
        #subet = subst + args.win_length
        #et = st + args.win_length
        subutt = "{utt}-{s:08d}-{e:08d}".format(utt=utt, s=round(100 * subst), e=round(100 * subet))
        print('%s\t%s\t%.2lf\t%.2lf' % (subutt, reco, st, et))
    else:
        if et - st >= args.min_length:
            subutt = "{utt}-{s:08d}-{e:08d}".format(utt=utt, s=round(100 * subst), e=round(100 * subet))
            print('%s\t%s\t%.2lf\t%.2lf' % (subutt, reco, st, et))



