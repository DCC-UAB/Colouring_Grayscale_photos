from torch import nn
#from ..Preprocessing.DataClass import *
#from ..Preprocessing.LoaderClass import *

class modelAutoencoder1(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 512, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 256, 3, stride=1, padding=1)
        )
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 512, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 512, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 256, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 128, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 2, 4, stride=2, padding=1),
            nn.Tanh()
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
