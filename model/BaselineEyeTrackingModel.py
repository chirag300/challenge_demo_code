import torch
import torch.nn as nn

class CNN_GRU(nn.Module):
    def __init__(self, args):
        super().__init__() 
        self.args = args
        self.conv1 = nn.Conv2d(args.n_time_bins, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.lstm = nn.LSTM(input_size=144768, hidden_size=512, num_layers=3, batch_first=True)
        self.dropout = nn.Dropout(0.5) 
        self.fc = nn.Linear(512, 2)

    def forward(self, x):
        batch_size, seq_len, channels, height, width = x.shape
        x = x.view(batch_size * seq_len, channels, height, width)
        x = x.permute(0, 1, 3, 2)

        x = self.conv1(x)
        x = torch.relu(x)
        x = self.conv2(x)
        x = torch.relu(x)
        x = self.conv3(x)
        x = torch.relu(x)
        x = self.conv4(x)
        x = torch.relu(x)
        x = self.conv5(x)
        x = torch.relu(x)

        x = self.pool(x)

        x = x.view(batch_size, seq_len, -1)
        x, _ = self.lstm(x)

        # Unpack the LSTM output
        x = torch.unbind(x, dim=1)

        # Apply dropout separately to each time step
        x = [self.dropout(step) for step in x]

        # Stack the results back into a tensor
        x = torch.stack(x, dim=1)

        x = self.fc(x)
        return x
