import numpy as np
from tqdm import trange
import sys
import random
import hydra
from omegaconf import DictConfig, OmegaConf
import os
import networkx as nx
from matplotlib import pyplot as plt
import math
from tqdm import tqdm

import torch.optim as optim
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.autograd import grad
from torch.distributions import Categorical
from torch_geometric.data import HeteroData, Data
from torch_geometric.nn import to_hetero
from torch_geometric.data import Batch

from mushroom_rl.algorithms.actor_critic import SAC, DDPG
from mushroom_rl.core import Core, Logger
from mushroom_rl.environments.gym_env import Gym
from mushroom_rl.utils.dataset import compute_J, parse_dataset

from basenet.tasks.grasp_sequence import GraspSequence
from basenet.networks.grasp_sequence import PolicyNet

device = 'cuda:0'

np.random.seed()

no_of_objs = 10

def create_actor_graph(state):
    data = HeteroData()

    data["robot"].x = torch.tensor([[state[0], state[1], state[2]]])

    # object to grasp
    objs_to_grasp = []
    states = []
    j = 0
    for idx in range(0, no_of_objs):
        i = (idx * 4) + 4
        if state[i+3] > 0:
            states.append(state[i:i+3].float().tolist())
            objs_to_grasp.append(j)
            j = j + 1

    data["object"].x = torch.FloatTensor(states)

    if len(objs_to_grasp) > 0:
        data["robot", "grasps", "object"].edge_index = torch.tensor([np.zeros((len(objs_to_grasp),)), np.array(objs_to_grasp)], dtype=torch.long)
        data["object", "grasps", "robot"].edge_index = torch.tensor([np.array(objs_to_grasp), np.zeros((len(objs_to_grasp),))], dtype=torch.long)
    else:
        data["object"].x = torch.empty(1,3,dtype=torch.long)
        data["robot", "grasps", "object"].edge_index = torch.empty(2,0,dtype=torch.long)
        data["object", "grasps", "robot"].edge_index = torch.empty(2,0,dtype=torch.long)


    source_es = []
    destination_es = []

    for si in range(0, len(objs_to_grasp)):
        for di in range(0, len(objs_to_grasp)):
            if si is not di:
                source_es.append(si)
                destination_es.append(di)

    if len(objs_to_grasp) > 1:     
        data["object", "surrounds", "object"].edge_index = torch.tensor([np.array(source_es), np.array(destination_es)], dtype=torch.long)
    else:
        data["object", "surrounds", "object"].edge_index = torch.empty(2,0,dtype=torch.long)

    if len(objs_to_grasp) > 0: 
        data["object", "loops", "object"].edge_index = torch.tensor([np.array(objs_to_grasp),np.array(objs_to_grasp)], dtype=torch.long)
    else:
        data["object", "loops", "object"].edge_index = torch.empty(2,0,dtype=torch.long)

    data["robot", "loops", "robot"].edge_index = torch.tensor([[0], [0]], dtype=torch.long)

    return data.to(device='cuda:0')


class ReinforceLoss(nn.Module):
    def __init__(self):
        super(ReinforceLoss, self).__init__()

    def forward(self, probs, reward, action_idx):
        # print("Probs:", probs)
        action_distribution = Categorical(probs)
        loss = -action_distribution.log_prob(action_idx) * reward
        return loss


def generate_episode(mdp, policy_net, optimizer, device="cuda:0", max_episode_len = no_of_objs):
    status = False
    state = mdp.reset()
    ep_length = 0

    while not status:
        ep_length+=1
        state = torch.from_numpy(state).to(device='cuda:0')
        state_g = create_actor_graph(state)
        dl = []
        dl.append(state_g)
        batch = Batch.from_data_list(dl)
        # print("Batch:", batch.x_dict)

        action = torch.zeros(no_of_objs,).to(device=device) # max objects
        action_probs = policy_net(batch.x_dict, batch.edge_index_dict)['object'].squeeze()

        if action_probs.numel() == 1:
            action[0] = 1
            action_probs = action_probs.float()
            action_probs = action_probs.unsqueeze(0)
            action_probs = torch.softmax(action_probs, dim=-1)
        else:
            action_probs = torch.softmax(action_probs, dim=-1)
            action_distribution = Categorical(action_probs)
            action_idx = action_distribution.sample()            
            action[action_idx] = 1

        
        b_reward = mdp.get_baseline_reward()
        
        next_state, reward, status, __ = mdp.step(action)

        new_episode_sample = (state, action, reward, b_reward), action_probs
        yield new_episode_sample

        state = next_state
        if ep_length > max_episode_len:
            return

    # Add the final state, action, and reward for reaching the exit position
    new_episode_sample = (state, None, 0, 0), None
    yield new_episode_sample
   



def train(cfg, n_epochs, n_batch):
    np.random.seed()

    # MDP
    mdp = GraspSequence(cfg)
    use_cuda = torch.cuda.is_available()

    prior_agent_path = os.path.join(cfg.task.train.pre_trained_agent_dir, '{}.msh'.format(cfg.task.train.pre_trained_agent_name))
    print("Prior agent path:", prior_agent_path)
    prior_agent = SAC.load(prior_agent_path)

    mdp.load_prior_agent(prior_agent)

    policy_net = PolicyNet(3, 256, 1).to(device='cuda:0')
    data = create_actor_graph(torch.tensor([1.,1.,1.,0.,1.,1.,1.,1.,0.,0.,0.,1.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,
                                                1.,1.,1.,1.,0.,0.,0.,1.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.]).to(device='cuda:0'))
    policy_net = to_hetero(policy_net, data.metadata(), aggr='sum').to(device='cuda:0')


    loss_fn = ReinforceLoss()

    # print("Ep:", list(episode))

    
    gamma = 0.95
    lr_policy_net = 1e-4
    optimizer = torch.optim.Adam(policy_net.parameters(), lr=lr_policy_net)

    save_rewards = np.zeros((1000,))
    last_idx = 0

    for epoch_no in tqdm(range(n_epochs)):
        optimizer.zero_grad() 
        loss = 0
        lengths = []

        for batch_no in range(n_batch):
            episode = list(generate_episode(mdp, policy_net, optimizer, device=device))
            lengths.append(len(episode))
            
            rewards = []
            b_rewards = []

            for t, ((state, action, reward, b_reward), probs) in enumerate(episode[:-1]):
                rewards.append(reward)
                b_rewards.append(b_reward)
            rewards = torch.FloatTensor(rewards).to(device='cuda:0')
            b_rewards = torch.FloatTensor(b_rewards).to(device='cuda:0')

            # print("Learned rewards:", torch.sum(-rewards)/1000, " Baseline rewards:", torch.sum(-b_rewards)/1000)
            
            for t, ((state, action, reward, b_reward), probs) in enumerate(episode[:-1]):
                gammas_vec = gamma ** (torch.arange(t+1, len(episode))-t-1).to(device='cuda:0')
                G = torch.sum(gammas_vec * rewards[t:len(episode)-1])
                G_baseline = torch.sum(gammas_vec * b_rewards[t:len(episode)-1])
                action_idx = torch.argmax(action)
                loss = loss + loss_fn(probs, G-G_baseline, action_idx)
        
        loss = loss/(n_batch*no_of_objs)    
        loss.backward()
        optimizer.step()

        torch.save(policy_net, '{}/policy_model.pth'.format(cfg.task.test.save_dir))

        if epoch_no % 50  == 0:
            rs = np.zeros(200,)
            for en in tqdm(range(200)):
                status = False
                state = mdp.reset()
                reward_ep = 0
                while not status:
                    state = torch.from_numpy(state).to(device='cuda:0')
                    state_g = create_actor_graph(state)
                    dl = []
                    dl.append(state_g)
                    batch = Batch.from_data_list(dl)
                    # print("Batch:", batch.x_dict)

                    action = torch.zeros(no_of_objs,).to(device=device) # max objects
                    action_probs = policy_net(batch.x_dict, batch.edge_index_dict)['object'].squeeze()

                    if action_probs.numel() == 1:
                        action[0] = 1
                        action_probs = action_probs.float()
                        action_probs = action_probs.unsqueeze(0)
                        action_probs = torch.softmax(action_probs, dim=-1)
                    else:
                        action_probs = torch.softmax(action_probs, dim=-1)
                        action_distribution = Categorical(action_probs)
                        action_idx = action_distribution.sample()            
                        action[action_idx] = 1

                    # print("Action:", action)

                    next_state, reward, status, __ = mdp.step(action)
                    state = next_state
                    reward_ep = reward_ep + reward
                rs[en] = reward_ep
            print("Avergae reward after ", epoch_no, " epochs:", np.mean(rs))

            save_rewards[last_idx] = np.mean(rs)
            np.savez('{}/data_logs2.npz'.format(cfg.task.test.save_dir), full_save=True, J=save_rewards)
            last_idx = last_idx + 1


def test(cfg, n_episodes):
    np.random.seed()

    # MDP
    mdp = GraspSequence(cfg)
    use_cuda = torch.cuda.is_available()

    prior_agent_path = os.path.join(cfg.task.train.pre_trained_agent_dir, '{}.msh'.format(cfg.task.train.pre_trained_agent_name))
    print("Prior agent path:", prior_agent_path)
    prior_agent = SAC.load(prior_agent_path)

    mdp.load_prior_agent(prior_agent)

    policy_net = torch.load('{}/policy_model.pth'.format(cfg.task.test.save_dir))


    states_s = np.array([])
    rewards_s = np.array([])
    status_s = np.array([])
    actions_s = np.array([])

    for episode_num in tqdm(range(n_episodes)):
        status = False
        state = mdp.reset()
        
        while not status:
            state = torch.from_numpy(state).to(device='cuda:0')
            state_g = create_actor_graph(state)
            dl = []
            dl.append(state_g)
            batch = Batch.from_data_list(dl)
            # print("Batch:", batch.x_dict)

            states_s = np.append(states_s, np.expand_dims(state.cpu().detach().numpy(), axis=0))

            action = torch.zeros(10,).to(device=device) # max objects
            action_probs = policy_net(batch.x_dict, batch.edge_index_dict)['object'].squeeze()

            if action_probs.numel() == 1:
                action[0] = 1
                action_probs = action_probs.float()
                action_probs = action_probs.unsqueeze(0)
                action_probs = torch.softmax(action_probs, dim=-1)
            else:
                action_probs = torch.softmax(action_probs, dim=-1)
                action_distribution = Categorical(action_probs)
                action_idx = action_distribution.sample()            
                action[action_idx] = 1

            # print("Action:", action)

            actions_s = np.append(actions_s, np.expand_dims(action.cpu().detach().numpy(), axis=0))

            next_state, reward, status, __ = mdp.step(action)
            state = next_state

            rewards_s = np.append(rewards_s, np.expand_dims(reward, axis=0))
            status_s = np.append(status_s, np.expand_dims(status, axis=0))

        
            np.savez('{}/data_for_analysis.npz'.format(cfg.task.test.save_dir), full_save=True, 
                states=states_s, actions=actions_s, rewards=rewards_s, status=status_s)




@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg : DictConfig) -> None:
    config = OmegaConf.to_yaml(cfg)
    print(config)
    train(cfg, 50000, 10)
    

if __name__ == '__main__':
    main()