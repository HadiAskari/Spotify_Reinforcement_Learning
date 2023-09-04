import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, seq_len=5, num_features=26):
        super().__init__()
        
        # length of sequence
        self.seq_len = seq_len
        
        # first lstm layer
        self.lstm1 = nn.LSTM(
            input_size=num_features,
            hidden_size=24,
            num_layers=1,
            batch_first=True
        )
        
        # second lstm layer
        self.lstm2 = nn.LSTM(
            input_size=24,
            hidden_size=8,
            num_layers=1,
            batch_first=True
        )
        
    def forward(self, x):
        # passthru first lstm
        x, (_, _) = self.lstm1(x)
        
        # passthru second lstm
        x, (hidden_state, cell_state) = self.lstm2(x)
        
        # return latent representation
        return hidden_state.reshape(-1, 1, 8)
    
class Decoder(nn.Module):
    def __init__(self, seq_len=5, num_features=26):
        super().__init__()
        
        # length of sequence
        self.seq_len = seq_len
        
        # first lstm layer
        self.lstm1 = nn.LSTM(
            input_size=8,
            hidden_size=24,
            num_layers=1
        )
        
        # linear layer for reconstruction
        self.fc = nn.Linear(24, num_features)
        
    def forward(self, x):
        # repeat hidden state
        x = x.repeat(1, self.seq_len, 1)

        # passthru first lstm
        x, (_, _) = self.lstm1(x)
        
        # passthru linear
        return self.fc(x)
    

class AutoEncoder(nn.Module):
    def __init__(self, seq_len, num_features):
        super().__init__()
        
        self.encoder = Encoder(seq_len, num_features)
        self.decoder = Decoder(seq_len, num_features)
        
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded