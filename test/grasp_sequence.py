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

            actions_s = np.append(actions_s, np.expand_dims(action.cpu().detach().numpy(), axis=0))

            next_state, reward, status, __ = mdp.step(action)
            state = next_state

            rewards_s = np.append(rewards_s, np.expand_dims(reward, axis=0))
            status_s = np.append(status_s, np.expand_dims(status, axis=0))

        
            # np.savez('{}/data_for_analysis.npz'.format(cfg.task.test.save_dir), full_save=True, 
            #     states=states_s, actions=actions_s, rewards=rewards_s, status=status_s)




@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg : DictConfig) -> None:
    config = OmegaConf.to_yaml(cfg)
    print(config)
    test(cfg, 50000)
    

if __name__ == '__main__':
    main()