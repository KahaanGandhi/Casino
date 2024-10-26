import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm
import pandas as pd

class BaseModel:
    def train_model(self, train_loader, epochs, learning_rate, device):
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.to(device)
        self.train()
        
        # Outer progress bar for epochs
        with tqdm(total=epochs, desc="Training Progress", position=0, leave=False) as epoch_pbar:
            for epoch in range(epochs):
                epoch_loss = 0
                # Inner progress bar for batch processing
                with tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}", leave=False, position=1) as batch_pbar:
                    for seq, labels in train_loader:
                        seq, labels = seq.to(device), labels.to(device)
                        optimizer.zero_grad()
                        y_pred = self(seq)
                        loss = criterion(y_pred.view(-1), labels.view(-1))
                        loss.backward()
                        optimizer.step()
                        epoch_loss += loss.item()
                        # Update batch progress bar with loss
                        batch_pbar.set_postfix(loss=epoch_loss / (batch_pbar.n + 1))
                        batch_pbar.update(1)
                # Update outer epoch progress bar
                epoch_pbar.update(1)
        self.eval()

    def predict(self, test_seq, device):
        self.eval()
        with torch.no_grad():
            pred = self(test_seq.to(device))
        return pred.cpu().numpy()

class RNNModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, output_size=1):
        super(RNNModel, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        
        # Define optimizer and loss function here
        self.optimizer = optim.Adam(self.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()

    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.fc(out[:, -1, :])
        return out

    def predict(self, x, device):
        self.eval()
        with torch.no_grad():
            x = x.to(device)
            out = self(x)
        return out.cpu().numpy()

class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, output_size=1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        
        # Define optimizer and loss function here
        self.optimizer = optim.Adam(self.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

    def predict(self, x, device):
        self.eval()
        with torch.no_grad():
            x = x.to(device)
            out = self(x)
        return out.cpu().numpy()


class MomentumMeanReversionModel:
    def __init__(self, momentum_lookback, ma_window):
        self.momentum_lookback = momentum_lookback
        self.ma_window = ma_window

    def generate_signals(self, data):
        momentum = data.pct_change(periods=self.momentum_lookback)
        moving_average = data.rolling(window=self.ma_window).mean()
        mean_reversion = data - moving_average

        signals = pd.DataFrame(index=data.index)
        signals['Signal'] = 0
        buy_signal = (momentum > 0) & (mean_reversion < 0)
        sell_signal = (momentum < 0) & (mean_reversion > 0)
        signals.loc[buy_signal, 'Signal'] = 1
        signals.loc[sell_signal, 'Signal'] = -1
        signals['Signal'] = signals['Signal'].fillna(method='ffill').fillna(0)
        return signals['Signal']
