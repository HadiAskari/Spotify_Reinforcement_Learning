import torch
import pandas as pd
import os
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import numpy as np
from encoder_model import *
from dataset import SpotifyDataset

# set torch device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# read data
DATA_DIR = '../data'
sessions = pd.read_csv(os.path.join(DATA_DIR, 'processed', 'training.csv'))
features = torch.load(os.path.join(DATA_DIR, 'processed', 'track_features.pt')).to(device).float()

# create cache for reading sessions
cache = {}
def get_session_sequence(sessions, session_id, seq_len=5):
    if session_id not in cache:
        cache[session_id] = sessions[sessions['session_id'] == session_id].iloc[:seq_len]['track_id'].tolist()
    return features[cache[session_id]]

# sample 10000 data points from spotify sessions
training_data = SpotifyDataset(sessions, 100000)

# Dataloader initialization
loader = torch.utils.data.DataLoader(training_data, batch_size = 128, shuffle = True)

# Model Initialization (5 sequences of 26 features each)
model = AutoEncoder(5, 26).to(device)

# Validation using MSE Loss function
loss_function = torch.nn.MSELoss()

# Using an Adam Optimizer with lr = 0.1
optimizer = torch.optim.Adam(model.parameters(),
                         lr = 1e-1,
                         weight_decay = 1e-8)

# train model for 2000 epochs
epochs = 2000
outputs = []
losses = []

for epoch in tqdm(range(epochs)):
    
    epoch_loss = []
    for session_ids in loader:
        
        # load data from batch
        batch = []
        for session_id in session_ids:
            feat = get_session_sequence(sessions, session_id)
            batch.append(feat)
            
        batch = torch.stack(batch).to(device)
        
        # Output of Autoencoder
        enc, dec = model(batch)

        # Calculating the loss function
        loss = loss_function(dec, batch)        
        
        # The gradients are set to zero,
        # the gradient is computed and stored.
        # .step() performs parameter update
        optimizer.zero_grad()
        loss.backward()        
        optimizer.step()

        print(loss, end='\r')
        # Storing the losses in a list for plotting
        epoch_loss.append(loss.cpu().detach().numpy())
        
    losses.append(np.mean(epoch_loss))


# save trained model
torch.save(model.state_dict(), 'models/autoencoder.pt')
torch.save(model.encoder.state_dict(), 'models/encoder.pt')

# show loss
plt.plot(losses)
plt.show()