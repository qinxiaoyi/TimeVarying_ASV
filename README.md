# TimeVarying_ASV

## Xiaoyi Qin, Na Li, Shufei Duan, Ming Li

This repository is our recent work, Investigaing Long-term and Short-term Temporal Variations in Speaker Verification.

The voice aging can decline the performance of speaker verification system. However, limited research has been conducted on time-varying speaker verification due to the absence of appropriate datasets. This paper aims to investigate the impact of long-term and short-term temporal variations in speaker verification and proposes solutions to mitigate these effects. 

For long-term speaker verification (i.e., cross-age speaker verification，CA-SV), we introduce an age-decoupling adversarial learning method to learn age-invariant speaker representation by mining age information from the VoxCeleb dataset. The source code and data resources of CA-SV are available on [github](https://github.com/qinxiaoyi/Cross-Age_Speaker_Verification).

For short-term speaker verification, we collect the SMIIP-TimeVarying (SMIIP-TV) Dataset, which records 373 speakers recording multiple time slots every day for 90 consecutive days and contains meta information. Using this dataset, we analyze the temporal variations of speech signals and propose a novel but realistic time-varying speaker verification task, termed incremental sequence-pair speaker verification. This task involves continuous interaction between enrollment audios and sequentially testing audios with the aim of improving performance over time. We introduce the template updating method to counter the negative effects over time, and then we formulate the template updating processing as a Markov Decision Process and propose a template updating method based on deep reinforcement learning (DRL). The policy network of DRL is treated as an agent to determine whether to update the template and how much the weights should be updated. This repository provide the source code and data resources of short-term speaker verification.

Note: This paper has been submitted to T-ASLP.

# Dataset application

Once the publication is accepted by the journal, we will make the database open source. If you wish to access the database now, you can apply and temporarily download it from [Zenodo](https://zenodo.org/).


# Implementation

## Step.0 Requirement

### Create env  

Base environment

```shell
conda create --name ISSV python=3.8 # create env
conda install jupyter notebook # most inference scripts are implemented on Jupyter for convenient visualization.
python -m ipykernel install --user --name ISSV --display-name ISSV
```

Training/inference enieonment.

```
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
pip -r requeirement.txt

# install gym_examples for create our own environment

cd ./gym_examples
python setup.py install
```


## Step.1 Preparing speaker embedding

In this section, you will need to use your pre-trained model to extract speaker embeddings. However, we have already prepared the embeddings we used in the experiment for you, in order to facilitate the replication of our results. The extracted embeddings are stored as a dictionary in the .npy format, where each audio corresponds to a vector.

```python
import glob
import numpy as np

embds_dict={}
for npy_path in glob.glob('./DRL-TU/egs/exp/embed/time_varying_all_T_epoch21_rank*.npy'):
    print(npy_path)
    embds_dict_tmp = np.load(npy_path,allow_pickle=True).item()
    embds_dict ={**embds_dict,**embds_dict_tmp}
```

## Step.2 Training of DRL-TU-MH

### 1. pretraining for  DRL-TU-MH (optional)

```shell
pretrain_DRL-TU-MH.ipynb
```

### 2. finetuning for DRL-TU

```shell
python ppo_train_multihead.py
```



## Step.3 Inference and evaluation

### 1. inference

```shell
inference.ipynb
```

### 2. visualization

```shell
Visual analysis.ipynb

```


# Detail of DRL-MH-TU

## Environment

## Pipeline


# Other 

## Task introduction

## Reference
