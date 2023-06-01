'''
Filter the target file by source file.
e.g.:
    python filter.py a.txt 0 b.txt 1 > filter_b.txt
    If an element in the column of target file is not in the column of the source file,
    remove it.

'''

import os
import sys

assert len(sys.argv) == 5, 'Usage: source file, column, target file, column'

s_file, sc, t_file, tc = sys.argv[1:]
sc, tc = int(sc), int(tc)

source_set = []
with open(s_file, 'r') as f:
    for line in f.readlines():
        source_set.append(line.split()[sc])
source_set = set(source_set)

target_content = []
with open(t_file, 'r') as f:
    for line in f.readlines():
        if line.split()[tc] in source_set:
            print(line, end="") # since there is a '\n' at the end of line
