import sys
import torch  
import gymnasium as gym
import numpy as np  
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np
import gym_examples
import gymnasium
import os
import time
# import pandas as pd
import random 

def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module

class AddBias(nn.Module):
    def __init__(self, bias):
        super(AddBias, self).__init__()
        self._bias = nn.Parameter(bias.unsqueeze(1))

    def forward(self, x):
#         if x.dim() == 2:
#             bias = self._bias.t().view(1, -1)
#         else:
#             bias = self._bias.t().view(1, -1, 1, 1)
        bias = self._bias.t().view(1, -1)
        return x + bias

#Categorical
class FixedCategorical(torch.distributions.Categorical):
    def sample(self):
        return super().sample().unsqueeze(-1)

    def log_probs(self, actions):
        return super().log_prob(actions.squeeze(-1)).view(actions.size(0), -1).sum(-1).unsqueeze(-1)

    def entropy(self):
        p = self.probs.masked_fill(self.probs <= 0, 1)
        return -1 * p.mul(p.log()).sum(-1)

    def mode(self):
        return self.probs.argmax(dim=-1, keepdim=True)

class Categorical(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(Categorical, self).__init__()

        init_ = lambda m: init(
            m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), gain=0.01
        )

        self.linear = init_(nn.Linear(num_inputs, num_outputs))

    def forward(self, x, mask=None,temperature=1):
        x = F.softmax(self.linear(x)/temperature,dim=-1)
        if mask is not None:
            return FixedCategorical(logits=x + torch.log(mask))
        else:
            return FixedCategorical(logits=x)

#Normal
class FixedNormal(torch.distributions.Normal):
    def log_probs(self, actions):
        return super().log_prob(actions).sum(-1, keepdim=True)

    def entropy(self):
        return super().entropy().sum(-1)

    def mode(self):
        return self.mean

class DiagGaussian(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(DiagGaussian, self).__init__()

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0))

        self.fc_mean = init_(nn.Linear(num_inputs, num_outputs))
#         desired_init_log_std = -0.693471 #exp(..) ~= 0.5
        desired_init_log_std = -1.609437 #exp(..) ~=0.2
#         desired_init_log_std = -2.302585 #exp(..) ~=0.1
        
        self.logstd = AddBias(desired_init_log_std * torch.ones(num_outputs)) #so no state-dependent sigma

    def forward(self, x, mask=None):
        action_mean = self.fc_mean(x)
#         print('action_mean',action_mean.shape,x.shape)
        zeros = torch.zeros(action_mean.size())
        if x.is_cuda:
            zeros = zeros.cuda()

        action_logstd = self.logstd(zeros)
        return FixedNormal(action_mean, action_logstd.exp())

class ActionHead(nn.Module):
    def __init__(self, input_dim, output_dim, type="categorical"):
        super(ActionHead, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.type = type
        if type == "categorical":
            self.distribution = Categorical(num_inputs=input_dim, num_outputs=output_dim)
        elif type == "normal":
            self.distribution = DiagGaussian(num_inputs=input_dim, num_outputs=output_dim)
        else:
            raise NotImplementedError

    def forward(self, input, mask):
        if self.type == "normal":
            return self.distribution(input)
        else:
            return self.distribution(input, mask)
        
class Pi_net(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Pi_net, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, int(hidden_size/2))
        self.action_heads = nn.ModuleList()
        self.action_heads.append(ActionHead(int(hidden_size/2), 2, type='categorical'))
        self.action_heads.append(ActionHead(int(hidden_size/2)+1, 1, type='normal'))
        
    def forward(self, s, deterministic=False):
        x = F.relu(self.linear1(s))
        x = F.relu(self.linear2(x))
        
        action_outputs=[]
        head_outputs=[]
        head_outputs.append(x)
        action_type_dist = self.action_heads[0](x,mask=None)
        if deterministic:
            action_type = action_type_dist.mode()
        else:
            action_type = action_type_dist.sample()
            
        head_outputs.append(action_type)
        action_outputs.append(action_type)
        head_output = torch.cat(head_outputs, dim=-1)

        head_dist = self.action_heads[1](head_output,mask=None)
        
        if deterministic:
            head_action = head_dist.sample()
        else:
            head_action = head_dist.rsample()
        
        action_outputs.append(head_action)

        joint_action_log_prob = action_type_dist.log_probs(action_type)
        entropy = action_type_dist.entropy().mean()
        
        joint_action_log_prob += head_dist.log_probs(head_action)

        entropy += head_dist.entropy().mean()
        action_outputs = torch.cat(action_outputs,dim=-1)
        return action_outputs, joint_action_log_prob, entropy

class V_net(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(V_net,self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, int(hidden_size/2))
        self.linear3 = nn.Linear(int(hidden_size/2), output_size)
    
    def forward(self, s, a):       
        if len(s.shape)==3:
            s = s.squeeze(1)
#         print(s.shape,a.shape) a[action type,action value]
        x = torch.cat([s, a], 1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x
    
class Actor_Critic(nn.Module):
    def __init__(self, input_size, hidden_size, output_size,start_episodes=0,pretrain=False):
        super(Actor_Critic,self).__init__()
        self.v = V_net(input_size+2, input_size, output_size)        
        self.pi = Pi_net(input_size, input_size)

        # pretrain
        if pretrain:
            pretrain_chp=torch.load('./exp/DRL-TU-MH_pretrained/pretrain_v1.pkl',
                                    map_location='cpu')
            self.pi.load_state_dict(pretrain_chp['model'],strict=False)
            self.pi.action_heads[0].distribution.linear.weight.data=pretrain_chp['model']['linear3.weight']
            self.pi.action_heads[0].distribution.linear.bias.data=pretrain_chp['model']['linear3.bias']
            self.pi.action_heads[1].distribution.fc_mean.weight.data=pretrain_chp['model']['linear4.weight']
            self.pi.action_heads[1].distribution.fc_mean.bias.data=pretrain_chp['model']['linear4.bias']
        
        # cour learning
        if start_episodes !=0:
            checkpoint=torch.load('./exp/ppo_multihead_v1/model_%s.pkl'%start_episodes,
                                    map_location='cpu')
            self.pi.load_state_dict(checkpoint['pi'])
            self.v.load_state_dict(checkpoint['v'])


def save_checkpoint(chk_dir, episode, actor_critic):
    torch.save({'pi': actor_critic.pi.state_dict(),'v': actor_critic.v.state_dict()}, os.path.join(chk_dir, 'model_%d.pkl' % episode))

class Agent(object):
    def __init__(self,start_episodes=0,pretrain=False):
        self.actor_critic=Actor_Critic(256, 256, 1,start_episodes=start_episodes,pretrain=pretrain)
        self.old_pi = Pi_net(256, 256)        #旧策略网络
        self.old_v = V_net(256+2, 256, 1)     #旧价值网络 
        if pretrain:
            pretrain_chp=torch.load('./exp/DRL-TU-MH_pretrained/pretrain_v1.pkl',
                                    map_location='cpu')
            self.old_pi.load_state_dict(pretrain_chp['model'],strict=False)
            self.old_pi.action_heads[0].distribution.linear.weight.data=pretrain_chp['model']['linear3.weight']
            self.old_pi.action_heads[0].distribution.linear.bias.data=pretrain_chp['model']['linear3.bias']
            self.old_pi.action_heads[1].distribution.fc_mean.weight.data=pretrain_chp['model']['linear4.weight']
            self.old_pi.action_heads[1].distribution.fc_mean.bias.data=pretrain_chp['model']['linear4.bias']    
            
        if start_episodes !=0:
            checkpoint=torch.load('./exp/test_version//model_%s.pkl'%start_episodes,
                                map_location='cpu')
            self.old_pi.load_state_dict(checkpoint['pi'])
            self.old_v.load_state_dict(checkpoint['v'])
            
        self.data = []               #用于存储经验
        self.step = 0
        self.value_loss_coef = 0.5
        self.entropy_coef = 0.05
        self.optimizer = torch.optim.Adam(self.actor_critic.parameters(), lr=LR)
        
 
    def choose_action(self, s):
        with torch.no_grad():
            action_outputs, joint_action_log_prob, entropy = self.old_pi(s,deterministic=True)
        return action_outputs
 
    def push_data(self, transitions):
        self.data.append(transitions)
 
    def sample(self):
        l_s, l_a, l_r, l_s_, l_done = [], [], [], [], []
        for item in self.data:
            s, a, r, s_, done = item
            l_s.append(torch.tensor(s, dtype=torch.float))
            l_a.append(a)
            l_r.append(torch.tensor([[r]], dtype=torch.float))
            l_s_.append(torch.tensor(s_, dtype=torch.float))
            l_done.append(torch.tensor([[done]], dtype=torch.float))
        s = torch.cat(l_s, dim=0)
        a = torch.cat(l_a, dim=0)
        r = torch.cat(l_r, dim=0)
        s_ = torch.cat(l_s_, dim=0)
        done = torch.cat(l_done, dim=0)
        self.data = []
        return s, a, r, s_, done
 
    def updata(self):
        self.step += 1
        s, a, r, s_, done = self.sample()
        for _ in range(K_epoch):
            with torch.no_grad():
                '''loss_v'''
                td_target = r + GAMMA * self.old_v(s_,a) * (1 - done)
#                 print(td_target.shape,r.shape,self.old_v(s_,a).shape,done.shape)
                '''loss_pi'''
                action_outputs, joint_action_log_prob, entropy = self.old_pi(s,deterministic=True)
                
                log_prob_old = joint_action_log_prob
                
                td_error = r + GAMMA * self.actor_critic.v(s_,a) * (1 - done) - self.actor_critic.v(s,a)
                td_error = td_error.detach().numpy()
                A = []
                adv = 0.0
                for td in td_error[::-1]:
                    adv = adv * GAMMA * LAMBDA + td[0]
                    A.append(adv)
                A.reverse()
                A = torch.tensor(A, dtype=torch.float).reshape(-1, 1)
 
            action_outputs, joint_action_log_prob, entropy = self.actor_critic.pi(s)
            log_prob_new = joint_action_log_prob
        
            ratio = torch.exp(log_prob_new - log_prob_old)
            L1 = ratio * A
            L2 = torch.clamp(ratio, 1 - CLIP, 1 + CLIP) * A
            loss_pi = -torch.min(L1, L2).mean()
 
            loss_v = F.mse_loss(td_target.detach(), self.actor_critic.v(s,a))
 
            self.optimizer.zero_grad()
            (loss_v * self.value_loss_coef + loss_pi - entropy *self. entropy_coef).backward()
            self.optimizer.step()
            
        self.old_pi.load_state_dict(self.actor_critic.pi.state_dict())
        self.old_v.load_state_dict(self.actor_critic.v.state_dict())    


spk2utt={i.split()[0]:i.split()[1:] for i in open('../../data/train/spk2utt')}
utt2dur={i.split()[0]:i.split()[1] for i in open('../../data/dur.scp')}

from collections import defaultdict
spk2day2utt=defaultdict(list)
for spk in spk2utt:
    spk2day2utt[spk]=defaultdict(list)
    for utt in spk2utt[spk]:
        utt_time = utt.split('_')[1][:4]
        day=utt_time
        spk2day2utt[spk][day].append(utt)   
        

import glob
embds_dicts={}
for npy_path in glob.glob('./embed/time_varying_all_T_epoch21_rank*.npy'):
    print(npy_path)
    embds_dict = np.load(npy_path,allow_pickle=True).item()
    embds_dicts ={**embds_dicts,**embds_dict}

spk2utt=defaultdict(list)
for spk in spk2day2utt:
    for day in spk2day2utt[spk]:
        for utt in spk2day2utt[spk][day]:
            if float(utt2dur[utt])>0.64: 
                spk2utt[spk].append(utt)
                
spk2num = {spk:len(spk2utt[spk]) for spk in spk2utt}
spk2num_gt100 = [spk for spk in spk2num if spk2num[spk]>200] 

new_spk2day2utt={}
for spk in spk2num_gt100:
    new_spk2day2utt[spk]=spk2day2utt[spk]

    
os.environ['CUDA_VISIBLE_DEVICES']='3'    
chk_dir='./exp/ppo_multihead_v1/'

if not os.path.exists(chk_dir) :
    os.makedirs(chk_dir)

env = gymnasium.make('gym_examples/SpkembdUpdateEnv_multihead',spk2day2utt=new_spk2day2utt,embd_dict=embds_dicts,threshold_hard=0,threshold_deter=0.5)

env.reset()
# env.render()
# hyperparameters

LR = 2e-5

K_epoch = 5
GAMMA = 0.99
LAMBDA = 0.95
CLIP = 0.1

# Constants
num_steps = 500
pretrain=True
start_episodes =0
max_episodes = 500000

agent = Agent(start_episodes=start_episodes,pretrain=pretrain)
logs = open('%s/train.out' % chk_dir, 'w')

all_lengths = []
average_lengths = []
    
for episode in range(start_episodes,max_episodes):
    utt_per_day = random.sample([3,5,10],1)[0]
#     utt_per_day = 5
    state0, info = env.reset(utt_per_day = utt_per_day)
    episode_reward = 0
    start_time = time.time()
    for step in range(num_steps):
        action0 = agent.choose_action(torch.tensor(state0, dtype=torch.float))  
        action_info = [action0,info,step]
        state1, reward1, done, _ , info = env.step(action_info)
        agent.push_data((state0, action0, reward1, state1, done))   
        episode_reward += reward1 
        state0 = state1
        
        if done or step == num_steps-1:
            all_lengths.append(step)
            average_lengths.append(np.mean(all_lengths[-10:]))
            end_time = time.time()
            break
            
    agent.updata()
    
    if episode%1== 0:
        logs.write("episode: {}, reward: {:.3f}, eer: {:.3f}, total length: {}, determine_true: {},  true trial: {}, true update:{}, false trial: {}, false update:{}, days: {}, perutt_day: {},average length: {:.2f}, time: {:.2f}s \n".format(episode, episode_reward, info['eer'],step,info['determine_T'], info['num_true_trial'], info['true_update_num'], info['num_false_trial'], info['false_update_num'], info['days'], utt_per_day,average_lengths[-1], end_time-start_time))
        logs.flush()
                
    if episode%100==0:
            save_checkpoint(chk_dir, episode, agent.actor_critic)

