import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import torchviz

class MultiLabelAutoencoder(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MultiLabelAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 60),
            nn.LeakyReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(60, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 64),
            nn.LeakyReLU(),
            nn.Linear(64, output_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

features_df=pd.read_csv('po.csv', header=None)
labels_df=pd.read_csv('polabel.csv', header=None)

input_data = features_df.values
output_data = labels_df.values

input_tensor = torch.tensor(input_data, dtype=torch.float32)
output_tensor = torch.tensor(output_data, dtype=torch.float32)


