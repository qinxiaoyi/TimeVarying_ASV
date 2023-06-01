
# import numpy as np
# import kaldi_io
# import sys

# # this script is transform the npz to ark
# # npz_path is the npz path
# # wav_scp is the npz's wav.scp
# # embedding_ark_path is ark file of output embedding from npz file

# npz_path = sys.argv[1]
# wav_scp = sys.argv[2]
# embedding_ark_name = sys.argv[3]+'.{}'

# embedding = np.load(npz_path,allow_pickle=True)
# embd_dict = embedding['arr_0']

# utt_ids = [i.split()[0] for i in open(wav_scp)]
# num_per_ark=70000
# num_ark_file = int(np.ceil(len(utt_ids)/num_per_ark))

# for i in range(num_ark_file):
#     with open(embedding_ark_name.format(i)+'.ark','wb') as f:
#         for utt_id in utt_ids[i*num_per_ark:(i+1)*num_per_ark]:
#             kaldi_io.write_vec_flt(f, embd_dict.item().get(utt_id), key=utt_id)
#             #f.write(utt_id+' '+str(embd_dict.item().get(utt_id)[0]).replace('\n','').replace('[',' [ ').replace(']',' ]')+'\n')
            
            
import numpy as np
import kaldi_io
import sys
from multiprocessing import Pool    
import os    
import re

# this script is transform the npz to ark
# npz_path is the npz path
# wav_scp is the npz's wav.scp
# embedding_ark_path is ark file of output embedding from npz file

def npz2ark(num,embedding_ark_name,embd_dict,utt_ids,num_per_ark):
    print('run %s'% num)
    print(embedding_ark_name.format(num)+'.txt')
    with open(embedding_ark_name.format(num)+'.txt','w') as f:
        
        for i,utt_id in enumerate(utt_ids[num*num_per_ark:(num+1)*num_per_ark]):
            #kaldi_io.write_vec_flt(f, embd_dict.item().get(utt_id), key=utt_id)
            f.write(utt_id+' '+re.sub(' +', ' ', str(embd_dict.item().get(utt_id)[0]).replace('\n','')).replace('[',' [ ').replace(']',' ]')+'\n')
            if i %10000==0:
                print("Thread %s has processed %s"%(num,i))
            



npz_path = sys.argv[1]
wav_scp = sys.argv[2]

embd_dict = np.load(npz_path,allow_pickle=True)
utt_ids = [i.split()[0] for i in open(wav_scp)]

if len(sys.argv) < 5:
    embedding_ark_name = sys.argv[3]
    num_ark_file=1
    num_per_ark=len(utt_ids)
else:
    num_per_ark=int(sys.argv[4])
    embedding_ark_name = sys.argv[3]+'.{}'
    num_ark_file = int(np.ceil(len(utt_ids)/num_per_ark))

po = Pool(num_ark_file)

for i in range(num_ark_file):
        # async非堵塞添加,to_work为参数名,i为传递的参数,单个参数一定要加逗号!一定要加逗号!一定要加逗号!
        po.apply_async(npz2ark,(i,embedding_ark_name,embd_dict,utt_ids,num_per_ark))

        
print("----begin----")

# 关闭进程池,不再接收新的任务,开始执行任务
po.close()

# 主进程等待所有子进程结束
po.join()
print("----end----")
