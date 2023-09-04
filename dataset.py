from random import sample
from torch.utils.data import Dataset
from random import sample

class SpotifyDataset(Dataset):

    def __init__(self, sessions, num_sessions=None):
        self.session_ids = list(sessions['session_id'].unique())
        if num_sessions is not None:
            self.session_ids = sample(self.session_ids, num_sessions)

    def __len__(self):
        return len(self.session_ids)

    def __getitem__(self, idx):
        return self.session_ids[idx]