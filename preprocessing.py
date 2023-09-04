import os
import pandas as pd
import torch

# folder containing data
DATA_DIR = '../data'

def concat_features():
    # merge track features
    features = []
    for feat in os.listdir(os.path.join(DATA_DIR, 'track_features')):
        if not feat.endswith('.csv'):
            continue
        csv = os.path.join(DATA_DIR, 'track_features', feat)
        features.append(pd.read_csv(csv, encoding = "ISO-8859-1"))

    # write final features to csv
    pd.concat(features).to_csv(os.path.join(DATA_DIR, 'processed', 'features.csv'), index=False)


# concatenate features if not already
if not os.path.exists(os.path.join(DATA_DIR, 'processed', 'features.csv')):
    concat_features()

# write torch tensor for features
if not os.path.exists(os.path.join(DATA_DIR, 'processed', 'track_features.pt')):
    features = pd.read_csv(os.path.join(DATA_DIR, 'processed', 'features.csv'))
    features['mode'] = features['mode'].map(lambda x : ['major', 'minor'].index(x))
    features = features[['track_id', 'acousticness', 'beat_strength', 'bounciness', 'danceability',
        'dyn_range_mean', 'energy', 'flatness', 'instrumentalness', 'key',
        'liveness', 'loudness', 'mechanism', 'mode', 'organism', 'speechiness',
        'tempo', 'time_signature', 'valence', 'acoustic_vector_0',
        'acoustic_vector_1', 'acoustic_vector_2', 'acoustic_vector_3',
        'acoustic_vector_4', 'acoustic_vector_5', 'acoustic_vector_6',
        'acoustic_vector_7']].set_index('track_id')
    torch.save(torch.from_numpy(features.to_numpy()), (os.path.join(DATA_DIR, 'processed', 'track_features.pt')))


if not os.path.exists(os.path.join(DATA_DIR, 'processed', 'sessions.csv')):
    # sample sessions from dataset
    sessions = os.listdir(os.path.join(DATA_DIR, 'training_set'))

    ## read in data from log files
    dfs = []
    for session in sessions:
        if 'log_0' not in session:
            continue
        dfs.append(pd.read_csv(os.path.join(DATA_DIR, 'training_set', session)))
    df = pd.concat(dfs)

    # sample some sessions and write sequences
    sampled_sessions = set(df['session_id'].sample(1000000))

    ## create a new session sample df
    sample_df = df[df['session_id'].isin(sampled_sessions)][['session_id', 'track_id_clean', 'not_skipped']].copy()

    # write corresponding track id indexes for faster lookups
    ## caching for track ids
    mapping = {track_id: ind for ind, track_id in enumerate(features.index)}
    sample_df['track_id'] = sample_df['track_id_clean'].map(lambda x : mapping[x])

    ## write stripped file to disk
    sample_df[['session_id', 'track_id', 'not_skipped']].to_csv(os.path.join(DATA_DIR, 'processed', 'sessions.csv'), index=False)