import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class CriticNetwork(nn.Module):
    def __init__(self, input_shape, output_shape, n_features, **kwargs):
        super().__init__()

        n_input = input_shape[-1]
        n_output = output_shape[-1]

        self._h1 = nn.Linear(n_input, n_features)
        self._h2 = nn.Linear(n_features, 1* n_features)
        self._h3 = nn.Linear(1* n_features, 1* n_features)
        self._h7 = nn.Linear(n_features, n_output)

        nn.init.xavier_uniform_(self._h1.weight,
                                gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self._h2.weight,
                                gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self._h3.weight,
                                gain=nn.init.calculate_gain('relu'))                                                                                                                   
        nn.init.xavier_uniform_(self._h7.weight,
                                gain=nn.init.calculate_gain('linear'))

    def forward(self, state, action):
        # print("State:", state.shape)
        # print("Action:", action.shape)

        state_action = torch.cat((state.float(), action.float()), dim=1)
        features1 = F.relu(self._h1(state_action))
        # features1 = F.dropout(features1, p=0.5, training=self.training)
        features2 = F.relu(self._h2(features1))
        # features2 = F.dropout(features2, p=0.5, training=self.training)
        features3 = F.relu(self._h3(features2))
        # features3 = F.dropout(features3, p=0.5, training=self.training)
        q = self._h7(features3)
        # print("Q:", q)
        return torch.squeeze(q)


class ActorNetwork(nn.Module):
    def __init__(self, input_shape, output_shape, n_features, **kwargs):
        super(ActorNetwork, self).__init__()

        n_input = input_shape[-1]
        n_output = output_shape[-1]

        self._h1 = nn.Linear(n_input, n_features)
        self._h2 = nn.Linear(n_features, 1* n_features)
        self._h3 = nn.Linear(1* n_features, 1* n_features)
        self._h7 = nn.Linear(n_features, n_output)

        nn.init.xavier_uniform_(self._h1.weight,
                                gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self._h2.weight,
                                gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self._h3.weight,
                                gain=nn.init.calculate_gain('relu'))                                                                                                                    
        nn.init.xavier_uniform_(self._h7.weight,
                                gain=nn.init.calculate_gain('linear'))

    def forward(self, state):
        features1 = F.relu(self._h1(torch.squeeze(state, 1).float()))
        # features1 = F.dropout(features1, p=0.5, training=self.training)
        features2 = F.relu(self._h2(features1))
        # features2 = F.dropout(features2, p=0.5, training=self.training)
        features3 = F.relu(self._h3(features2))
        a = self._h7(features3)
        # print("Action in nw:", a)
        return a