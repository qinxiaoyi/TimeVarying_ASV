# TimeVarying_ASV

## Xiaoyi Qin, Na Li, Shufei Duan, Ming Li

This repository is our recent work, Investigating Long-term and Short-term Temporal Variations in Speaker Verification.

The performance of speaker verification systems can be adversely affected by time domain variations. However, limited research has been conducted on time-varying speaker verification due to the absence of appropriate datasets. This paper aims to investigate the impact of long-term and short-term temporal variations in speaker verification and proposes solutions to mitigate these effects. 

For long-term speaker verification (i.e., cross-age speaker verification), we introduce an age-decoupling adversarial learning method to learn age-invariant speaker representation by mining age information from the VoxCeleb dataset. The source code and data resources of CA-SV are available on [Cross-Age_Speaker_Verification](https://github.com/qinxiaoyi/Cross-Age_Speaker_Verification).

For short-term speaker verification, we collect the SMIIP-TimeVarying (SMIIP-TV) Dataset, which includes recordings at multiple time slots every day from 373 speakers for 90 consecutive days and other relevant meta information. Using this dataset, we analyze the temporal variations of speaker embeddings and propose a novel but realistic time-varying speaker verification task, termed incremental sequence-pair speaker verification. This task involves continuous interaction between enrollment audios and a sequence of testing audios with the aim of improving performance over time. We introduce the template updating method to counter the negative effects over time, and then formulate the template updating processing as a Markov Decision Process and propose a template updating method based on deep reinforcement learning (DRL). The policy network of DRL is treated as an agent to determine if and how much should the template be updated.

In summary, this paper releases our collected database and  investigates both the long-term and short-term speaker temporal variations and provides insights and solutions into time-varying speaker verification.

Note: This paper has been submitted to T-ASLP.

# Section.1 Dataset 

## SMIIP-TV dataset introduction
The SMIIP-TimeVarying Dataset (SMIIP-TV), is a speaker verification dataset designed for research purposes that focuses on short-term time-varying of speaker verification. The recordings language is Mandarin. The dataset contains recordings from **373** speakers who provided utterances over 90 consecutive days, in which each speaker needs to record multiple utterances at varying time slots in each day. To ensure that recording time spans the full day without location limitations, we developed an Android application, which randomly assigns recording tasks in five different time slots: 6:00-8:00, 9:00-11:00, 12:00-14:00, 17:00-19:00, and 20:00-22:00, as shown in Fig.\ref{fig:smiiptv_recordedtime}. In each time slot, speakers provide three utterances, including both text-dependent and text-independent speech samples. Table \ref{tab:content} shows the contents of the recording. Additional meta information such as speaker region (total 27 provinces, China), age, and cellphone type was collected. Additionally, speakers were asked to report details on their physical state (total 7 types, including normal, sleepy, eating, sore throat, exercise, cold/fever, others), recording environment (total 16 scenes), and the type of background noise (totally 4 types, including quiet, normal, noisy, extremely noisy), all were manually reviewed. The dataset statistics are presented in Fig.\ref{fig:smiiptv_statistic}. The majority of speakers in the dataset are college students and their families from Shanxi Province, China, and the gender distribution is balanced (171 males:202 females). Most recordings were made  indoors, with the majority of the noise and physical conditions being normal. Speakers were also encouraged to report various scenes with different physical conditions. Due to the challenge of continuously recording for 90 days, some speakers were unable to provide recordings for the entire duration. Finally, 133 speakers recorded for the entire 90-day period, and we selected 58 of them as the SMIIP-TV test set, and the remaining speaker data (315 speakers) is adopted as the training set.
![image](https://github.com/qinxiaoyi/TimeVarying_ASV/blob/main/img/smiiptv_age_dis.png "age")

![image](https://github.com/qinxiaoyi/TimeVarying_ASV/blob/main/img/smiiptv_body_dis.png)

![image](https://github.com/qinxiaoyi/TimeVarying_ASV/blob/main/img/smiiptv_noise_dis_m.png)

![image](https://github.com/qinxiaoyi/TimeVarying_ASV/blob/main/img/smiiptv_recordedtime_dis.png)

![image](https://github.com/qinxiaoyi/TimeVarying_ASV/blob/main/img/smiiptv_scene_dis.png)


## Data application

Once the publication is accepted by the journal, we will make the database open source. If you wish to access the database now, you can contact us at [ming.li369@duke.edu](ming.li369@duke.edu) or [xiaoyi.qin@dukekunshan.edu.cn](xiaoyi.qin@dukekunshan.edu.cn) to apply and temporarily download it.

# Section.2 Implementation

## Step.0 Requirement

### Create env  

Base environment

```shell
conda create --name ISSV python=3.8 # create env
conda install jupyter notebook # most inference scripts are implemented on Jupyter for convenient visualization.
python -m ipykernel install --user --name ISSV --display-name ISSV
```

Training/inference environment.

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
