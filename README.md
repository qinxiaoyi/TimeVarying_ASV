# TimeVarying_ASV

## Xiaoyi Qin, Na Li, Shufei Duan, Ming Li

This repository is our recent work, **Investigating Long-term and Short-term Temporal Variations in Speaker Verification**.

The performance of speaker verification systems can be adversely affected by time domain variations. However, limited research has been conducted on time-varying speaker verification due to the absence of appropriate datasets. This paper aims to investigate the impact of long-term and short-term temporal variations in speaker verification and proposes solutions to mitigate these effects. 

For long-term speaker verification (i.e., cross-age speaker verification, CASV), we introduce an age-decoupling adversarial learning method to learn age-invariant speaker representation by mining age information from the VoxCeleb dataset. The source code and data resources of CASV are available on [Cross-Age_Speaker_Verification](https://github.com/qinxiaoyi/Cross-Age_Speaker_Verification).

For short-term speaker verification, we collect the SMIIP-TimeVarying (SMIIP-TV) Dataset, which includes recordings at multiple time slots every day from 373 speakers for 90 consecutive days and other relevant meta information. Using this dataset, we analyze the temporal variations of speaker embeddings and propose a novel but realistic time-varying speaker verification task, termed incremental sequence-pair speaker verification. This task involves continuous interaction between enrollment audios and a sequence of testing audios with the aim of improving performance over time. We introduce the template updating method to counter the negative effects over time, and then formulate the template updating processing as a Markov Decision Process and propose a template updating method based on deep reinforcement learning (DRL). The policy network of DRL is treated as an agent to determine if and how much should the template be updated.

In summary, this paper releases our collected database and  investigates both the long-term and short-term speaker temporal variations and provides insights and solutions into time-varying speaker verification.

Note: This paper has been submitted to T-ASLP.

# Section.1 Dataset 

## SMIIP-TV dataset introduction
The SMIIP-TimeVarying Dataset (SMIIP-TV), is a speaker verification dataset designed for research purposes that focuses on short-term time-varying of speaker verification. The recordings language is **Mandarin**. The dataset contains recordings from **373** speakers who provided utterances over **90 consecutive days**, in which each speaker needs to record multiple utterances at varying time slots in each day. To ensure that recording time spans the full day without location limitations, we developed an Android application, which randomly assigns recording tasks in five different time slots:**6:00-8:00, 9:00-11:00, 12:00-14:00, 17:00-19:00, and 20:00-22:00**, as shown in the following figure.

<div align="center">
<img src="https://github.com/qinxiaoyi/TimeVarying_ASV/blob/main/img/smiiptv_recordedtime_dis.png" width=45%/>
</div>
    
In each time slot, speakers provide three utterances, including both text-dependent and text-independent speech samples. Following table shows the recording contents. 
    
<div align="center">
<img src="https://github.com/qinxiaoyi/TimeVarying_ASV/blob/main/img/content.png" width=45%/>
</div>

Additional meta information such as **speaker region (total 27 provinces, China)**, **age**, and **cellphone type** was collected.

<div align="center">
<img src="https://github.com/qinxiaoyi/TimeVarying_ASV/blob/main/img/smiiptv_age_dis.png" width=45%/>
</div>
    
Additionally, speakers were asked to report details on their **physical state (total 7 types, including normal, sleepy, eating, sore throat, exercise, cold/fever, others)**, **recording environment (total 16 scenes)**, and the type of **background noise (totally 4 types, including quiet, normal, noisy, extremely noisy)**, all were manually reviewed. The majority of speakers in the dataset are college students and their families from Shanxi Province, China, and the gender distribution is balanced (**171 males:202 females**).
<div align="center">
<img src="https://github.com/qinxiaoyi/TimeVarying_ASV/blob/main/img/smiiptv_body_dis.png" width="250"/><img src="https://github.com/qinxiaoyi/TimeVarying_ASV/blob/main/img/smiiptv_noise_dis_m.png" width="250"/><img src="https://github.com/qinxiaoyi/TimeVarying_ASV/blob/main/img/smiiptv_scene_dis.png" width="400"/>
</div>
Most recordings were made  indoors, with the majority of the noise and physical conditions being normal. Speakers were also encouraged to report various scenes with different physical conditions. Due to the challenge of continuously recording for 90 days, some speakers were unable to provide recordings for the entire duration. Finally, 133 speakers recorded for the entire 90-day period, and we selected 58 of them as the SMIIP-TV test set, and the remaining speaker data (315 speakers) is adopted as the training set.


## Data application

Once the publication is accepted by the journal, we will make the database open source, including audio files and the metadata mentioned above. If you wish to access the database now, you can contact us at [ming.li369@duke.edu](ming.li369@duke.edu) or [xiaoyi.qin@dukekunshan.edu.cn](xiaoyi.qin@dukekunshan.edu.cn) to apply and temporarily download it.

# Section.2 Implementation

The training and inference code will be made available after the paper is accepted. The code includes instructions on how to build the reinforcement learning environment, how to conduct training, and how to perform evaluations.
