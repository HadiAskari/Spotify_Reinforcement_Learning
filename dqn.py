import torch
import torch.nn as nn
import pandas as pd
import os
from random import sample, randint
from tqdm.auto import tqdm
import numpy as np
import matplotlib.pyplot as plt
from dataset import SpotifyDataset
from encoder_model import *

# load torch device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# set DATA_DIR here
DATA_DIR = '../data'

# load dataset
sessions = pd.read_csv(os.path.join(DATA_DIR, 'processed', 'sessions.csv'))
sessions['not_skipped'] = sessions['not_skipped'].map(lambda x : 1 if x else 0)
features = torch.load(os.path.join(DATA_DIR, 'processed', 'track_features.pt')).to(device).float()

# configuration for replay memory
replay_memory = {}
def get_session_sequence(sessions, session_id, seq_len=5):
    if session_id not in replay_memory:
        session = sessions[sessions['session_id'] == session_id]
        session_pos = randint(7, len(session))
        session = session.iloc[session_pos - 7: session_pos]
        replay_memory[session_id] = (session['track_id'].tolist(), session['not_skipped'].tolist())
    return features[replay_memory[session_id][0]], torch.tensor(replay_memory[session_id][1])

# dataloader for training data
training_data = SpotifyDataset(sessions, 100000)


# DQN model V1
class DQN(nn.Module):
    def __init__(self, encoder):
        super(DQN, self).__init__()
        
        self.encoder = encoder
        self.fc1 = nn.Linear(175, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 2)
        self.softmax = nn.Softmax(2)
        
    def forward(self, history, actions, next_song):
        # encode history and repeat
        x = self.encoder(history).repeat(1, len(history[0]), 1)
        # add actions to each vector
        x = torch.cat((x, actions.reshape(-1, 5, 1)), 2)
        # concat with next song features
        x = torch.cat((x, next_song.reshape(-1, 1, 26).repeat(1, 5, 1)), 2)
        # concat entire to single
        x = x.reshape(-1, x.shape[1] * x.shape[2])
        # fully connected linear layers
        x = self.fc1(x)
        x = self.fc2(x)
        return self.fc3(x)
    

# load saved encoder models
encoder1 = Encoder()
encoder1.load_state_dict(torch.load('models/encoder.pt'))

encoder2 = Encoder()
encoder2.load_state_dict(torch.load('models/encoder.pt'))


# Dataloader initialization
loader = torch.utils.data.DataLoader(training_data, batch_size = 128, shuffle = True)

# Model Initialization
policy_dqn = DQN(encoder1).to(device)
target_dqn = DQN(encoder2).to(device)

# Validation using MSE Loss function
loss_function = torch.nn.SmoothL1Loss()

# Using an Adam Optimizer with lr = 0.1
optimizer = torch.optim.AdamW(policy_dqn.parameters(),
                         lr = 1e-4,
                         amsgrad = True)

# train the DQN model
epochs = 2000
outputs = []
losses = []
rewards = []
TAU = 0.0025

for epoch in tqdm(range(epochs)):
    epoch_rewards = 0
    epoch_loss = []
    for session_ids in loader:
        
        batch = []
        batch_actions = []

        for session_id in session_ids:
            feat, not_skipped = get_session_sequence(sessions, session_id)
            batch.append(feat)
            batch_actions.append(not_skipped)          
            
        batch = torch.stack(batch).float().to(device)
        batch_actions = torch.stack(batch_actions).float().to(device)
        
        # current policy Q
        current_q = policy_dqn(batch[:,:5,:], batch_actions[:,:5], batch[:,5,:])
        max_q = current_q.max(1)[0]
        
        # target Q
        with torch.no_grad():
            next_q = target_dqn(batch[:,1:6,:], batch_actions[:,1:6], batch[:,6,:]).max(1)[0]
          
        # calculate reward for correct predictions
        reward = (batch_actions[:, 5] == current_q.argmax(1)).int()
                  
        # calculate expected value of q
        expected_q = reward + (next_q * 1)
        
        # Calculating the loss function
        loss = loss_function(max_q, expected_q)
        
        # The gradients are set to zero,
        # the gradient is computed and stored.
        # .step() performs parameter update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Storing the losses in a list for plotting
        epoch_loss.append(loss.cpu().detach().numpy())
        
        epoch_rewards += reward.sum()
    
    target_dqn.load_state_dict(policy_dqn.state_dict())
    
    losses.append(np.mean(epoch_loss))
    rewards.append(epoch_rewards.cpu())
    print(epoch, rewards[-1], end='\r')

plt.plot(rewards)
plt.show()