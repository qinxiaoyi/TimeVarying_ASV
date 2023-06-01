import numpy as np
import random
import torch
import gymnasium as gym
from scipy import spatial
from gymnasium import spaces
import itertools
from scipy import spatial
from .util import compute_eer
from collections import defaultdict

class SpkembdUpdateEnv_v0(gym.Env):

    def _chose_test_utt(self,target_spk,targe_prob=0.8):
        """
        目的：随机挑选测试音频
        参数：target_spk：注册说话人id，
             targe_prob: 注册说话人被选中概率（挑选正样本概率）
        """
        label = np.random.choice([1,0], p=[targe_prob,1-targe_prob])
        if label:
            test_utt = self._chose_target_utt(target_spk)
        else:
            test_utt = self._chose_nontarget_utt(target_spk)
            
        return test_utt,label
    
    def _chose_nontarget_utt(self,tar_spk):
        """
        目的：随机挑选负样本
        """
        test_spk = random.sample(list(self.spk2day2utt.keys()),1)[0]
        if test_spk != tar_spk:
            day = random.sample(list(self.spk2day2utt[test_spk].keys()),1)[0]
            test_utt = random.sample(self.spk2day2utt[test_spk][day],1)[0]
#             print(test_utt)
            return test_utt
        else:
            return self._chose_nontarget_utt(tar_spk)
        
    def _chose_target_utt(self,target_spk):
        """
        目的：随机挑选正样本
        """
#         print(target_spk,self.pre_spk_utts)
        test_utt = self.pre_spk_utts[0]
        self.pre_spk_utts = self.pre_spk_utts[1:]
        return test_utt
    
    def _chose_enroll_embd(self,tar_spk):
        """
        暂时弃用
        """
        test_spk = random.sample(self.spk2day2utt,1)[0]
        if test_spk != tar_spk:
            day = self.spk2day2utt[test_spk]
            test_utt = random.sample(self.spk2day2utt[test_spk][day],1)[0]
            return test_utt
        else:
            return self._chose_nontarget_utt(tar_spk)
    
class SpkembdUpdateEnv_multihead(SpkembdUpdateEnv_v0):
    def __init__(self, spk2day2utt, embd_dict,threshold_hard=0.65, threshold_deter=0.65, embd_dim=128, render_mode=None):
        print('SpkembdUpdateEnv_multihead')
        self.spk2day2utt = spk2day2utt
        self.embd_dict = embd_dict
        self.embd_dim = embd_dim
        self.threshold_hard =threshold_hard
        self.threshold_deter = threshold_deter
        
        self.observation_space = spaces.Box(-1, 1, shape=(embd_dim*2,), dtype=float)
        self.action_space = spaces.Box(0, 1, shape=(1,), dtype=float)
        
        self.spk2avgembd=defaultdict(list)
        for spk in spk2day2utt:
            for day in spk2day2utt[spk]:
                for utt in spk2day2utt[spk][day]:
                    utt_embd = self.embd_dict[utt]
                    self.spk2avgembd[spk].append(utt_embd)
        
        for spk in self.spk2avgembd:
            self.spk2avgembd[spk]=np.array(self.spk2avgembd[spk]).mean(0)
     
    def reset(self, seed=None, utt_per_day = 3):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)
        self.false_determine=0
        self.episode_spk = random.sample(list(self.spk2day2utt.keys()),1)[0]
        self.true_all = 0
        
        pre_spk_utts = []
        num_target_trial =0
        num_nontarget_trial =0
        num_chose_days=[]
        
        self.true_score=[]
        self.false_score=[]
        self.true_update_num=0
        self.false_update_num=0
        
        # chose enroll utt start
        exist_days = [num for num,i in enumerate(self.spk2day2utt[self.episode_spk])]
        enrol_day =  random.sample(exist_days[:int(len(exist_days)/4)],1)[0]
#         print(enrol_day,len(exist_days))
        for num,day in enumerate(self.spk2day2utt[self.episode_spk]):
            if num==enrol_day:
                enroll_utts = self.spk2day2utt[self.episode_spk][day]
            elif num<enrol_day:
                continue
            else:
                ifdetermine_day = np.random.choice([1,0], p=[0.7,1-0.7])
                if ifdetermine_day:
                    num_chose_days.append(day)
                    sample_list = self.spk2day2utt[self.episode_spk][day]
                    utts_one_day = random.sample(sample_list,utt_per_day if len(sample_list)>utt_per_day else len(sample_list))
                    pre_spk_utts.append(utts_one_day)
        
        # chose enroll utt end
        
        self.pre_spk_utts = list(itertools.chain(*pre_spk_utts))
        
        
        enrol_embd = np.zeros((1,self.embd_dim))
        
        for enrol_num,utt_tmp in enumerate(enroll_utts):
            enrol_embd += self.embd_dict[utt_tmp] 
            
        enrol_embd = enrol_embd/len(enroll_utts) # no norm
        
        flag=1
        while flag:
            test_utt, label = self._chose_test_utt(target_spk=self.episode_spk,targe_prob=0.5)
            test_embd = self.embd_dict[test_utt]
            cosine_sim = 1 - spatial.distance.cosine(enrol_embd, test_embd)
            if (cosine_sim > self.threshold_hard) and label==0:
                flag = 0 
                num_nontarget_trial +=1
            elif label==1:
                flag =0
                num_target_trial +=1
                
                if cosine_sim > self.threshold_deter:
                    self.true_all +=1
      
        observation = np.concatenate((enrol_embd,test_embd),axis=1)
        
        info={'spk':self.episode_spk,'label':label,'test_utt':test_utt,'enrol_utt':enroll_utts,
                            'test_embd':test_embd,'enrol_embd':enrol_embd, 'num_true_trial':num_target_trial,'num_false_trial':num_nontarget_trial,'days':len(num_chose_days),'cosine_sim':cosine_sim}
        
        return observation, info
    
    def step(self, action_info):
        action_output, info,num_steps = action_info[0],action_info[1],action_info[2]
        
        action_deter, action_value = action_output[0][0], action_output[0][1]
        
        label = info['label']
        enrol_embd = info['enrol_embd']
        test_embd = info['test_embd']
        num_target_trial =info['num_true_trial']
        num_nontarget_trial =info['num_false_trial']
        num_chose_days =info['days']
        reward = 0
        
        if action_deter==1:
            enrol_embd = (1-action_value)*enrol_embd + action_value*test_embd  
        
        # v2
        cosine_sim = 1 - spatial.distance.cosine(enrol_embd, test_embd)
        if label==1 and action_deter==1:
            reward += cosine_sim
        elif label==1 and action_deter==0:
            reward += cosine_sim
        elif label==0 and action_deter==0:
            pass
        elif label==0 and action_deter==1:
            reward -= cosine_sim
         
        # v1 + v2 
        # 1/0 + cosine enrol average
        cosine_sim = 1 - spatial.distance.cosine(enrol_embd, self.spk2avgembd[self.episode_spk])
        if label==1 and action_deter==1:
            reward += 0.5
            reward += cosine_sim
            self.true_update_num +=1
        elif label==1 and action_deter==0:
            reward += 0
            reward += cosine_sim
        elif label==0 and action_deter==0:
            reward += 0.5
            self.false_update_num +=1
        elif label==0 and action_deter==1:
            reward -= 1
        
        
        flag=1
        while flag:
#             print(target_spk,self.pre_spk_utts,terminated)
            test_utt, label = self._chose_test_utt(target_spk=self.episode_spk,targe_prob=0.5)
            test_embd = self.embd_dict[test_utt]
            cosine_sim = 1 - spatial.distance.cosine(enrol_embd, test_embd)
            if label==0:
                flag = 0 
                num_nontarget_trial +=1
                if cosine_sim > self.threshold_deter:
                    self.false_determine +=1
            elif label==1:
                flag = 0
                num_target_trial +=1
                if cosine_sim < self.threshold_deter:
                    self.false_determine +=1
        
        if label==1 :
            self.true_score.append(cosine_sim)
            if cosine_sim > self.threshold_deter+0.2:
                self.true_all +=1
        else:
            self.false_score.append(cosine_sim)
        
        terminated = (len(self.pre_spk_utts)==1) or (self.false_determine == 200) or num_steps==499
        eer=3.14
        if terminated: 
            eer, threshold_eer, mindct, threashold_dct = compute_eer(np.array(self.true_score), np.array(self.false_score))
        
        observation = np.concatenate((enrol_embd,test_embd),axis=1)
        
        info={'spk':self.episode_spk, 'label':label,'test_utt':test_utt,'test_embd':test_embd,'enrol_embd':enrol_embd,
              'num_true_trial':num_target_trial,'num_false_trial':num_nontarget_trial,'days':num_chose_days,'cosine_sim':cosine_sim,'determine_T':self.true_all,'eer':eer,'true_update_num':self.true_update_num,'false_update_num':self.false_update_num}
         
        return observation, reward, terminated, False, info