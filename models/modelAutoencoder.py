from torch import nn
import torch

class modelAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 4, stride=2, padding=1) #16, 64, 64
        self.pool1 = nn.MaxPool2d(2, stride=2, return_indices=True) #16, 32, 32
        self.conv2 = nn.Conv2d(16, 8, 3, stride=2, padding=1) #8, 16, 16
        self.pool2 = nn.MaxPool2d(2, stride=1, return_indices=True) #8, 14, 14
        self.upsample1 = nn.Upsample(scale_factor=2)
        self.conv3 = nn.Conv2d(8, 4, 3, stride=1, padding=1)

        self.unconv3 = nn.ConvTranspose2d(4, 8, 3, stride=1, padding=1)
        self.upsample2 = nn.Upsample(scale_factor=0.5)
        self.unpool1 = nn.MaxUnpool2d(2, stride=1)
        self.unconv1 = nn.ConvTranspose2d(8, 16, 4, stride=2, padding=1)
        self.unpool2 = nn.MaxUnpool2d(2, stride=2)
        self.unconv2 = nn.ConvTranspose2d(16, 2, 4, stride=2, padding=1)
        self.tanh = nn.Tanh()

    def encoder(self, x):
        x = torch.relu(self.conv1(x))
        x, self.indices1 = self.pool1(x)
        x = torch.relu(self.conv2(x))
        x, self.indices2 = self.pool2(x)
        return x

    def decoder(self, x):
        x = self.unpool1(x, self.indices2)
        x = torch.relu(self.unconv1(x))
        x = self.unpool2(x, self.indices1)
        x = self.unconv2(x)
        x = self.tanh(x)
        return x
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
