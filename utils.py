import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch
from torch.utils.data import Dataset

class StockDataset(Dataset):
    def __init__(self, sequences):
        self.sequences = sequences
    def __len__(self):
        return len(self.sequences)
    def __getitem__(self, idx):
        seq, label = self.sequences[idx]
        return torch.tensor(seq, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)

def load_data(tickers, start_date, end_date):
    data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
    data.ffill(inplace=True)
    data.dropna(inplace=True)
    return data

def create_sequences(data, lookback):
    sequences = []
    for i in range(len(data) - lookback):
        seq = data[i:i+lookback]
        label = data[i+lookback]
        sequences.append((seq, label))
    return sequences

def get_scaler(data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data.values.reshape(-1, 1))
    return scaler, scaled_data
